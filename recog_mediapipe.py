import os
import argparse
import cv2
import mediapipe as mp
import pandas    as pd
import numpy     as np

from pathlib import Path

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text  import Text



folder_path = '/Users/yutakanno/Library/CloudStorage/GoogleDrive-poohyuta604@gmail.com/My Drive/mediapipe_test'

model_path = Path( __file__ ).parent / 'pose_landmarker_heavy.task'

# 6台中2台で足りる前提 → 各カメラは厳しめに弾く
min_pose_detection_confidence = 0.8
min_pose_presence_confidence  = 0.8
min_tracking_confidence       = 0.8
visibility_threshold          = 0.9


def pose_recog( folder_path, video_name, model_path ):

    video_path = os.path.join( folder_path, f'{video_name}.mp4' )

    output_csv = f'{video_name}.csv'

    console = Console(legacy_windows=False)

    console.print(
        Panel(
            Text( 'MediaPipe Pose Landmark Extraction', style = 'bold cyan', justify = 'center' ),
            border_style = 'cyan',
            padding = ( 0, 2 ),
        )
    )


    # --------------------------------------------------
    # MediaPipe Pose (Tasks API)
    # --------------------------------------------------

    landmark_names = [ lm.name for lm in vision.PoseLandmark ]

    options = vision.PoseLandmarkerOptions(
        base_options = mp_tasks.BaseOptions( model_asset_path = str( model_path ) ),
        running_mode = vision.RunningMode.VIDEO,
        num_poses    = 1,
        min_pose_detection_confidence = min_pose_detection_confidence,
        min_pose_presence_confidence  = min_pose_presence_confidence,
        min_tracking_confidence       = min_tracking_confidence,
    )


    # --------------------------------------------------
    # Video
    # --------------------------------------------------

    cap = cv2.VideoCapture( video_path )

    fps       = cap.get( cv2.CAP_PROP_FPS )
    frame_num = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
    width     = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    height    = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    duration  = frame_num / fps if fps > 0 else 0

    info_table = Table( title = 'Video Info', show_header = False, border_style = 'dim' )
    info_table.add_column( 'Key', style = 'bold green' )
    info_table.add_column( 'Value', style = 'white' )
    info_table.add_row( 'Path', str( video_path ) )
    info_table.add_row( 'Resolution', f'{width} x {height}' )
    info_table.add_row( 'FPS', f'{fps:.2f}' )
    info_table.add_row( 'Frames', f'{frame_num:,}' )
    info_table.add_row( 'Duration', f'{duration:.1f} sec' )
    info_table.add_row( 'Model', model_path.name )
    info_table.add_row( 'Detection conf.', f'{min_pose_detection_confidence:.2f}' )
    info_table.add_row( 'Presence conf.', f'{min_pose_presence_confidence:.2f}' )
    info_table.add_row( 'Tracking conf.', f'{min_tracking_confidence:.2f}' )
    info_table.add_row( 'Visibility thresh.', f'{visibility_threshold:.2f}' )
    console.print( info_table )
    console.print()


    # --------------------------------------------------
    # Main Loop
    # --------------------------------------------------

    records = []

    frame_idx = 0
    detected_count = 0

    with vision.PoseLandmarker.create_from_options( options ) as landmarker:

        with Progress(
            SpinnerColumn(),
            TextColumn( '[bold blue]{task.description}' ),
            BarColumn( bar_width = 40 ),
            MofNCompleteColumn(),
            TextColumn( '•' ),
            TimeElapsedColumn(),
            TextColumn( '•' ),
            TimeRemainingColumn(),
            console = console,
            transient = False,
        ) as progress:

            task = progress.add_task( 'Processing frames', total = frame_num or None )

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )
                mp_image = mp.Image( image_format = mp.ImageFormat.SRGB, data = rgb )
                timestamp_ms = int( cap.get( cv2.CAP_PROP_POS_MSEC ) )

                result = landmarker.detect_for_video( mp_image, timestamp_ms )

                row = { 'frame': frame_idx }

                if result.pose_landmarks:

                    detected_count += 1
                    landmarks = result.pose_landmarks[ 0 ]

                    for i, lm in enumerate( landmarks ):

                        name = landmark_names[ i ]

                        # 閾値フィルタなし: visibility スコアに関わらず全て保存
                        row[ f'{name}_x' ] = lm.x
                        row[ f'{name}_y' ] = lm.y
                        row[ f'{name}_z' ] = lm.z
                        row[ f'{name}_v' ] = lm.visibility if lm.visibility is not None else np.nan

                else:

                    for name in landmark_names:

                        row[ f'{name}_x' ] = np.nan
                        row[ f'{name}_y' ] = np.nan
                        row[ f'{name}_z' ] = np.nan
                        row[ f'{name}_v' ] = np.nan

                records.append( row )
                progress.advance( task )

                frame_idx += 1

    cap.release()


    # --------------------------------------------------
    # DataFrame
    # --------------------------------------------------

    df = pd.DataFrame( records )
    detection_rate = ( detected_count / len( df ) * 100 ) if len( df ) else 0

    summary_table = Table( title = 'Summary', show_header = False, border_style = 'dim' )
    summary_table.add_column( 'Key', style = 'bold magenta' )
    summary_table.add_column( 'Value', style = 'white' )
    summary_table.add_row( 'Processed frames', f'{len( df ):,}' )
    summary_table.add_row( 'Pose detected', f'{detected_count:,}' )
    summary_table.add_row( 'Detection rate', f'{detection_rate:.1f}%' )
    summary_table.add_row( 'Columns', f'{len( df.columns ):,}' )
    summary_table.add_row( 'Output', output_csv )
    console.print()
    console.print( summary_table )
    console.print()

    preview_cols = [ 'frame', 'NOSE_x', 'NOSE_y', 'NOSE_z', 'NOSE_v', 'LEFT_WRIST_x', 'RIGHT_WRIST_x' ]
    preview_cols = [ col for col in preview_cols if col in df.columns ]
    console.print( '[bold yellow]Preview[/bold yellow]' )
    console.print( df[ preview_cols ].head() )

    df.to_csv( output_csv, index=False )

    console.print()
    console.print(
        f'[bold green]✓[/bold green] Saved [cyan]{output_csv}[/cyan] '
        f'([white]{len( df ):,}[/white] rows × [white]{len( df.columns ):,}[/white] columns)'
    )


def pose_recog_from_config( config: dict, model_path: Path ):
    """設定辞書からポーズ認識を実行する（複数動画セグメント対応）。"""

    import json as _json

    console = Console(legacy_windows=False)

    output_csv   = config.get( 'output_csv', 'output.csv' )
    synced_start = int( config.get( 'synced_start', 0 ) )
    det_conf     = float( config.get( 'det_conf',    0.8 ) )
    pres_conf    = float( config.get( 'pres_conf',   0.8 ) )
    track_conf   = float( config.get( 'track_conf',  0.8 ) )
    vis_thresh   = float( config.get( 'vis_thresh',  0.9 ) )
    segments     = config.get( 'video_segments', [] )
    save_to      = config.get( 'save_to', 'pose3d' )   # 'pose3d' or 'calib'

    console.print(
        Panel(
            Text( 'MediaPipe Pose (config mode)', style = 'bold cyan', justify = 'center' ),
            border_style = 'cyan',
            padding = ( 0, 2 ),
        )
    )
    console.print( f'[dim]Output CSV   : {output_csv}[/dim]' )
    console.print( f'[dim]Synced start : {synced_start}[/dim]' )
    console.print( f'[dim]Segments     : {len(segments)}[/dim]' )
    console.print()

    landmark_names = [ lm.name for lm in vision.PoseLandmark ]

    options = vision.PoseLandmarkerOptions(
        base_options = mp_tasks.BaseOptions( model_asset_path = str( model_path ) ),
        running_mode = vision.RunningMode.VIDEO,
        num_poses    = 1,
        min_pose_detection_confidence = det_conf,
        min_pose_presence_confidence  = pres_conf,
        min_tracking_confidence       = track_conf,
    )

    records = []
    output_frame_idx = synced_start
    global_timestamp_ms = 0

    with vision.PoseLandmarker.create_from_options( options ) as landmarker:

        for seg_i, seg in enumerate( segments ):

            video_path = seg[ 'path' ]
            seg_start  = int( seg.get( 'start', 0 ) )
            seg_end    = seg.get( 'end', None )

            cap = cv2.VideoCapture( video_path )
            if not cap.isOpened():
                console.print( f'[bold red]Cannot open:[/bold red] {video_path}' )
                continue

            fps        = cap.get( cv2.CAP_PROP_FPS ) or 30.0
            frame_num  = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
            if seg_end is None:
                seg_end = frame_num
            seg_end = min( seg_end, frame_num )

            console.print( f'[cyan]Segment {seg_i + 1}:[/cyan] {Path(video_path).name}  frames {seg_start}–{seg_end}' )

            # セグメント単位で出力フレームインデックスを上書きできる（ランダムフレーム用）
            seg_frame_index = seg.get( 'frame_index', None )
            cur_output_idx  = seg_frame_index if seg_frame_index is not None else output_frame_idx

            # スキップ
            cap.set( cv2.CAP_PROP_POS_FRAMES, seg_start )
            local_idx = seg_start

            with Progress(
                SpinnerColumn(),
                TextColumn( '[bold blue]{task.description}' ),
                BarColumn( bar_width = 40 ),
                MofNCompleteColumn(),
                TextColumn( '•' ),
                TimeElapsedColumn(),
                TextColumn( '•' ),
                TimeRemainingColumn(),
                console = console,
                transient = False,
            ) as progress:

                task = progress.add_task( f'Seg {seg_i + 1}', total = max(0, seg_end - seg_start) )

                while local_idx < seg_end:

                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )
                    mp_image = mp.Image( image_format = mp.ImageFormat.SRGB, data = rgb )

                    # タイムスタンプは連続して増加させる必要がある
                    timestamp_ms = global_timestamp_ms + int( local_idx * 1000.0 / fps )
                    result = landmarker.detect_for_video( mp_image, timestamp_ms )

                    row = { 'frame': cur_output_idx }

                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[ 0 ]
                        for i, lm in enumerate( landmarks ):
                            name = landmark_names[ i ]
                            # 閾値フィルタなし: visibility スコアに関わらず全て保存
                            row[ f'{name}_x' ] = lm.x
                            row[ f'{name}_y' ] = lm.y
                            row[ f'{name}_z' ] = lm.z
                            row[ f'{name}_v' ] = lm.visibility if lm.visibility is not None else np.nan
                    else:
                        for name in landmark_names:
                            row[ f'{name}_x' ] = np.nan
                            row[ f'{name}_y' ] = np.nan
                            row[ f'{name}_z' ] = np.nan
                            row[ f'{name}_v' ] = np.nan

                    records.append( row )
                    cur_output_idx  += 1
                    output_frame_idx += 1
                    local_idx += 1
                    progress.advance( task )

            # 次のセグメントのタイムスタンプがこのセグメント終端より大きくなるよう進める
            global_timestamp_ms += int( seg_end * 1000.0 / fps ) + 1
            cap.release()

    df = pd.DataFrame( records )

    # キャリブレーション用: 人が未検出のフレーム（全 _v が NaN）を除去
    if save_to == 'calib':
        _v_cols = [ c for c in df.columns if c.endswith( '_v' ) ]
        if _v_cols:
            _detected_mask = df[ _v_cols ].notna().any( axis=1 )
            _n_all = len( df )
            df = df[ _detected_mask ].reset_index( drop=True )
            console.print(
                f'[dim]人検出フレーム: {len(df)} / {_n_all} '
                f'(未検出 {_n_all - len(df)} フレームを除去)[/dim]'
            )

    # CSV は出力せず sync_config.json に直接書き込む
    # save_to='pose3d' → pose3d.landmarks、'calib' → calib.landmarks
    cam_name = config.get( 'cam_name' )
    if cam_name:
        try:
            from pathlib import Path as _Path2
            _cfg_p = _Path2( 'sync_config.json' )
            _cfg = _json.loads( _cfg_p.read_text( encoding='utf-8' ) ) if _cfg_p.exists() else {}
            if save_to not in _cfg:
                _cfg[ save_to ] = {}
            if 'landmarks' not in _cfg[ save_to ]:
                _cfg[ save_to ][ 'landmarks' ] = {}
            # カラム名 + データ配列（列指向で保存）
            _cfg[ save_to ][ 'landmarks' ][ cam_name ] = {
                'columns': df.columns.tolist(),
                'data':    df.values.tolist(),
            }
            _cfg_p.write_text( _json.dumps( _cfg, indent=2, ensure_ascii=False ), encoding='utf-8' )
            console.print()
            console.print(
                f'[bold green]✓[/bold green] Saved [cyan]{cam_name}[/cyan] landmarks '
                f'([white]{len(df):,}[/white] frames) → [cyan]sync_config.json[/cyan][dim][{save_to}][/dim]'
            )
        except Exception as e:
            console.print( f'[red]ERROR saving to sync_config.json: {e}[/red]' )
    else:
        console.print( '[yellow]cam_name not set — landmarks not saved[/yellow]' )


def parse_video_names( argv ):

    names = []

    for arg in argv:

        if arg.startswith( '--' ):
            names.append( arg[ 2: ] )
        else:
            raise argparse.ArgumentTypeError( f'unrecognized argument: {arg}' )

    return names


def resolve_video_paths( folder_path, video_names = None ):

    folder = Path( folder_path )

    if video_names:

        video_paths = []

        for name in video_names:

            path = folder / f'{name}.mp4'

            if not path.is_file():
                raise FileNotFoundError( path )

            video_paths.append( path )

        return video_paths

    return sorted( folder.glob( '*.mp4' ) )


if __name__ == '__main__':

    import json as _json

    parser = argparse.ArgumentParser(
        description = 'Extract pose landmarks from videos in folder_path.',
        epilog = (
            'Examples:\n'
            '  python recog_mediapipe.py              # all .mp4 files\n'
            '  python recog_mediapipe.py --cam1       # cam1.mp4 only\n'
            '  python recog_mediapipe.py --config cam1_config.json'
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument( '--config', metavar = 'PATH', default = None,
                         help = 'JSON config file for segment-based processing' )
    args, remaining = parser.parse_known_args()

    console = Console(legacy_windows=False)

    if args.config:
        # config モード
        cfg_path = Path( args.config )
        if not cfg_path.exists():
            console.print( f'[bold red]Config not found:[/bold red] {cfg_path}' )
            raise SystemExit( 1 )
        config = _json.loads( cfg_path.read_text( encoding = 'utf-8' ) )
        pose_recog_from_config( config, model_path )

    else:
        # 従来モード
        try:
            video_names = parse_video_names( remaining )
            video_paths = resolve_video_paths( folder_path, video_names or None )
        except argparse.ArgumentTypeError as e:
            parser.error( str( e ) )
        except FileNotFoundError as e:
            console.print( f'[bold red]Video not found:[/bold red] {e}' )
            raise SystemExit( 1 )

        if not video_paths:

            console.print( f'[bold red]No .mp4 files found in[/bold red] {folder_path}' )
            raise SystemExit( 1 )

        title = 'Selected' if video_names else 'Batch Processing'
        console.print(
            Panel(
                Text( f'{len( video_paths )} video(s)', style = 'bold white', justify = 'center' ),
                title = f'[bold cyan]{title}[/bold cyan]',
                border_style = 'cyan',
                padding = ( 0, 2 ),
            )
        )
        console.print()

        for i, video_path in enumerate( video_paths, start = 1 ):

            console.rule( f'[bold cyan]{i}/{len( video_paths )}[/bold cyan]  {video_path.name}' )
            console.print()

            pose_recog( folder_path, video_path.stem, model_path )

            if i < len( video_paths ):
                console.print()