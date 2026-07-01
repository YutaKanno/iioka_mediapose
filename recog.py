import os
import cv2
import mediapipe as mp
import pandas    as pd
import numpy     as np
import argparse

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


# --------------------------------------------------
# デフォルト設定
# --------------------------------------------------
# folder_path と video_name はコマンドライン引数で上書きする想定。
# ここでのデフォルト値は「ローカルでの動作確認用」の値にしてある。

default_folder_path = './data'
default_video_name  = 'cam2'

model_path = Path( __file__ ).parent / 'pose_landmarker_heavy.task'

# 6台中2台で足りる前提 → 各カメラは厳しめに弾く
min_pose_detection_confidence = 0.8
min_pose_presence_confidence  = 0.8
min_tracking_confidence       = 0.8
visibility_threshold          = 0.9


def pose_recog( folder_path, video_name, model_path, use_gpu ):

    video_path = os.path.join( folder_path, f'{video_name}.mp4' )

    output_csv = f'{video_name}.csv'

    console = Console()

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

    # Colab の GPU ランタイムで実行する場合は delegate に GPU を指定する。
    # ローカルの CPU 環境ではエラーになることがあるため、use_gpu フラグで切り替える。
    if use_gpu:
        delegate = mp_tasks.BaseOptions.Delegate.GPU
    else:
        delegate = mp_tasks.BaseOptions.Delegate.CPU

    base_options = mp_tasks.BaseOptions(
        model_asset_path = str( model_path ),
        delegate         = delegate,
    )

    options = vision.PoseLandmarkerOptions(
        base_options = base_options,
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
    info_table.add_row( 'Delegate', 'GPU' if use_gpu else 'CPU' )
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

                        if lm.visibility is None or lm.visibility < visibility_threshold:

                            row[ f'{name}_x' ] = np.nan
                            row[ f'{name}_y' ] = np.nan
                            row[ f'{name}_z' ] = np.nan
                            row[ f'{name}_v' ] = np.nan

                        else:

                            row[ f'{name}_x' ] = lm.x
                            row[ f'{name}_y' ] = lm.y
                            row[ f'{name}_z' ] = lm.z
                            row[ f'{name}_v' ] = lm.visibility

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


def parse_args():

    # Colab から `!python recog.py --folder_path ... --video_name ...` の形で
    # 呼び出せるように、パスと動画名をコマンドライン引数として受け取る。

    parser = argparse.ArgumentParser(
        description = 'MediaPipe Pose Landmarker を使って動画から姿勢データを抽出する'
    )

    parser.add_argument(
        '--folder_path',
        type = str,
        default = default_folder_path,
        help = '動画ファイルが置かれているフォルダのパス（例: Drive をマウントしたパス）',
    )

    parser.add_argument(
        '--video_name',
        type = str,
        default = default_video_name,
        help = '拡張子を除いた動画ファイル名（例: cam2 → cam2.mp4 を読み込む）',
    )

    parser.add_argument(
        '--cpu',
        action = 'store_true',
        help = '指定すると GPU ではなく CPU で実行する（ローカルでの動作確認用）',
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    pose_recog(
        folder_path = args.folder_path,
        video_name  = args.video_name,
        model_path  = model_path,
        use_gpu     = not args.cpu,
    )