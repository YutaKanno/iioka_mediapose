"""
三角測量のみ実行: camera_calibration.npz + camN.csv → landmarks_3d.csv

Usage:
    python triangulate_only.py [--config sync_config.json]

sync_config.json に必要なキー:
    calib.npz_path       - camera_calibration.npz へのパス
    active_cameras       - 使用するカメラリスト (例: ["cam1","cam2","cam3","cam4","cam6"])
    pose3d.start_frame   - 開始フレーム (synced)
    pose3d.end_frame     - 終了フレーム (synced)
    pose3d.vis_thresh    - 可視度閾値 (default: 0.5)
"""

import sys
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rich.console import Console

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io as _io
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
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
from rich.text import Text

console = Console(legacy_windows=False)

# ── ジオメトリヘルパー（est_camera_poses.py より複製）──────────

def projection_matrix( f, R, t, cx, cy ):
    K = np.array( [ [ f, 0, cx ], [ 0, f, cy ], [ 0, 0, 1 ] ] )
    return K @ np.hstack( [ R, t.reshape( 3, 1 ) ] )


def triangulate_multiview( obs_list, f_arr, rvecs, tvecs, cx, cy ):
    """
    obs_list: [(cam_idx, u_px, v_px), ...]
    f_arr   : 各カメラの焦点距離配列（カメラインデックスは npz 内の全カメラ順）
    rvecs   : (n_cam, 3) Rodriguez ベクトル
    tvecs   : (n_cam, 3)
    """
    A = []
    for ci, u, v in obs_list:
        R, _ = cv2.Rodrigues( rvecs[ ci ] )
        P = projection_matrix( f_arr[ ci ], R, tvecs[ ci ], cx, cy )
        A.append( u * P[ 2 ] - P[ 0 ] )
        A.append( v * P[ 2 ] - P[ 1 ] )

    _, _, Vt = np.linalg.svd( np.array( A ) )
    Xh = Vt[ -1 ]

    if abs( Xh[ 3 ] ) < 1e-12:
        return None

    return Xh[ :3 ] / Xh[ 3 ]


# ── メイン ──────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser(
        description = '三角測量のみ実行: camera_calibration.npz + camN.csv → landmarks_3d.csv',
    )
    parser.add_argument( '--config', metavar = 'PATH', default = 'sync_config.json',
                         help = 'sync_config.json へのパス (default: sync_config.json)' )
    args = parser.parse_args()

    cfg_path = Path( args.config )
    if not cfg_path.exists():
        console.print( f'[bold red]Config not found:[/bold red] {cfg_path}' )
        raise SystemExit( 1 )

    cfg = json.loads( cfg_path.read_text( encoding = 'utf-8' ) )

    # ── 設定読み込み ──────────────────────────────────────

    npz_path      = Path( cfg.get( 'calib', {} ).get( 'npz_path', 'camera_calibration.npz' ) )
    active_cameras = cfg.get( 'active_cameras', [] )
    pose3d        = cfg.get( 'pose3d', {} )
    start_frame   = int( pose3d.get( 'start_frame', 0 ) )
    end_frame     = pose3d.get( 'end_frame', None )
    default_thresh = float( pose3d.get( 'vis_thresh', 0.5 ) )
    # カメラ別閾値（なければ default_thresh を使用）
    cam_thresholds  = pose3d.get( 'cam_thresholds', {} )
    # landmarks は sync_config.json に直接格納（CSV不要）
    landmarks_store = pose3d.get( 'landmarks', {} )
    end_frame_val   = end_frame if end_frame is not None else 'all'
    output_csv      = f'3d_{start_frame}_{end_frame_val}.csv'

    console.print(
        Panel(
            Text( 'Triangulate Only', style = 'bold cyan', justify = 'center' ),
            border_style = 'cyan',
            padding = ( 0, 2 ),
        )
    )

    cfg_table = Table( title = 'Config', show_header = False, border_style = 'dim' )
    cfg_table.add_column( 'Key',   style = 'bold green' )
    cfg_table.add_column( 'Value', style = 'white' )
    cfg_table.add_row( 'NPZ',           str( npz_path ) )
    cfg_table.add_row( 'Active cams',   str( active_cameras ) )
    cfg_table.add_row( 'Start frame',   str( start_frame ) )
    cfg_table.add_row( 'End frame',     str( end_frame ) if end_frame is not None else '(all)' )
    cfg_table.add_row( 'Vis thresh',    str( cam_thresholds ) if cam_thresholds else str( default_thresh ) )
    cfg_table.add_row( 'Output CSV',    output_csv )
    console.print( cfg_table )
    console.print()

    # ── キャリブレーション読み込み ──────────────────────────

    if not npz_path.exists():
        console.print( f'[bold red]NPZ not found:[/bold red] {npz_path}' )
        raise SystemExit( 1 )

    data = np.load( npz_path, allow_pickle = True )
    all_cam_names = list( data[ 'camera_names' ] )
    f_arr  = data[ 'f' ]
    rvecs  = data[ 'rvecs' ]
    tvecs  = data[ 'tvecs' ]
    image_width  = int( data[ 'image_width' ] )
    image_height = int( data[ 'image_height' ] )
    cx = image_width  / 2.0
    cy = image_height / 2.0

    console.print( f'[dim]Loaded NPZ: {len(all_cam_names)} cameras  image={image_width}x{image_height}[/dim]' )

    # アクティブカメラのグローバルインデックスを解決
    if not active_cameras:
        active_cameras = all_cam_names
        console.print( '[yellow]active_cameras not set → using all cameras[/yellow]' )

    cam_indices = []
    for name in active_cameras:
        if name in all_cam_names:
            cam_indices.append( ( name, all_cam_names.index( name ) ) )
        else:
            console.print( f'[yellow]Camera {name} not in NPZ, skipping[/yellow]' )

    if not cam_indices:
        console.print( '[bold red]No valid cameras found.[/bold red]' )
        raise SystemExit( 1 )

    # ── ランドマーク読み込み（sync_config.json の pose3d.landmarks から）──

    mp_dataframes: dict[ str, pd.DataFrame ] = {}

    for cam_name, ci in cam_indices:
        lm_entry = landmarks_store.get( cam_name )
        if not lm_entry:
            console.print( f'[yellow]{cam_name}: landmarks not found in sync_config.json, skipping[/yellow]' )
            continue
        df = pd.DataFrame( lm_entry[ 'data' ], columns = lm_entry[ 'columns' ] )
        df = df[ df[ 'frame' ] >= start_frame ]
        if end_frame is not None:
            df = df[ df[ 'frame' ] <= end_frame ]
        df = df.reset_index( drop = True )
        mp_dataframes[ cam_name ] = ( ci, df )
        console.print( f'[dim]{cam_name}: {len(df)} rows (frames {start_frame}–{end_frame})[/dim]' )

    if not mp_dataframes:
        console.print( '[bold red]No CSV data loaded.[/bold red]' )
        raise SystemExit( 1 )

    # ランドマーク名（最初のCSVから取得）
    first_df = next( iter( mp_dataframes.values() ) )[ 1 ]
    landmark_names = [ col[ :-2 ] for col in first_df.columns if col.endswith( '_v' ) ]

    # すべてのフレーム（最初のアクティブカメラを基準）
    all_frames = first_df[ 'frame' ].values

    console.print()
    console.print( f'Frames: [cyan]{len(all_frames):,}[/cyan]  Landmarks: [cyan]{len(landmark_names)}[/cyan]' )
    console.print()

    # ── 三角測量 ────────────────────────────────────────

    records = []

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
    ) as progress:

        task = progress.add_task( 'Triangulating frames', total = len( all_frames ) )

        for frame in all_frames:

            for landmark in landmark_names:

                obs = []

                for cam_name, ( ci, df ) in mp_dataframes.items():

                    row = df[ df[ 'frame' ] == frame ]
                    if len( row ) == 0:
                        continue

                    vis = row[ f'{landmark}_v' ].values[ 0 ]
                    thresh = float( cam_thresholds.get( cam_name, default_thresh ) )
                    if pd.isna( vis ) or vis < thresh:
                        continue

                    u_px = row[ f'{landmark}_x' ].values[ 0 ] * image_width
                    v_px = row[ f'{landmark}_y' ].values[ 0 ] * image_height
                    obs.append( ( ci, u_px, v_px ) )

                if len( obs ) < 2:
                    continue

                X = triangulate_multiview( obs, f_arr, rvecs, tvecs, cx, cy )
                if X is None:
                    continue

                records.append( {
                    'frame':    int( frame ),
                    'landmark': landmark,
                    'x': X[ 0 ],
                    'y': X[ 1 ],
                    'z': X[ 2 ],
                    'n_views': len( obs ),
                } )

            progress.advance( task )

    # ── 保存 ─────────────────────────────────────────────

    if not records:
        console.print( '[yellow]Warning: 三角測量結果が空です。[/yellow]' )
        raise SystemExit( 1 )

    df_out = pd.DataFrame( records )
    df_out.to_csv( output_csv, index = False, encoding='utf-8' )

    tri_table = Table( title = 'Triangulation Result', show_header = False, border_style = 'dim' )
    tri_table.add_column( 'Key',   style = 'bold green' )
    tri_table.add_column( 'Value', style = 'white' )
    tri_table.add_row( 'Output',    output_csv )
    tri_table.add_row( 'Rows',      f'{len(df_out):,}' )
    tri_table.add_row( 'Frames',    f'{df_out["frame"].nunique():,}' )
    tri_table.add_row( 'Landmarks', f'{df_out["landmark"].nunique()}' )
    console.print()
    console.print( tri_table )
    console.print()
    console.print( f'[bold green]✓[/bold green] Saved [cyan]{output_csv}[/cyan]' )


if __name__ == '__main__':
    main()
