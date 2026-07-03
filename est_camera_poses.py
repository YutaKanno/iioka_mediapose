import sys
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io as _io
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

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
from rich.text import Text

console = Console(legacy_windows=False)

np.random.seed( 42 )

camera_names = [ 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6' ]

n_cam   = len( camera_names )
ref_cam = 3  # このカメラを原点として固定（cam4 = バックネット正面）

image_width, image_height = 640, 360

cx, cy = image_width / 2, image_height / 2

# ワンド形状（ローカル座標, x 軸方向 0.5 m 刻み）
wand_local = np.array(
    [ [ 0.0, 0.0, 0.0 ], [ 0.5, 0.0, 0.0 ], [ 1.0, 0.0, 0.0 ], [ 1.5, 0.0, 0.0 ] ]
)
n_wand_k = len( wand_local )

v_thresh           = 0.95
frame_step         = 10    # BA に使う MediaPipe フレームのサンプリング間隔
cam_reg_weight     = 0.05
huber_f_scale      = 500.0  # 収束前は大きめに（ほぼ全残差を quadratic 領域に）
max_tri_frames     = None  # 三角測量の最大フレーム数（None で無制限）


# --------------------------------------------------
# Geometry helpers
# --------------------------------------------------

def look_at_rvec_tvec( cam_pos, target = np.zeros( 3 ), up = np.array( [ 0.0, 1.0, 0.0 ] ) ):

    forward = target - cam_pos
    forward /= np.linalg.norm( forward )

    right = np.cross( forward, up )

    if np.linalg.norm( right ) < 1e-8:
        up = np.array( [ 0.0, 0.0, 1.0 ] )
        right = np.cross( forward, up )

    right /= np.linalg.norm( right )
    up_cam = np.cross( right, forward )

    R = np.stack( [ right, -up_cam, forward ], axis = 0 )
    t = -R @ cam_pos
    rvec, _ = cv2.Rodrigues( R )

    return rvec.ravel(), t


def projection_matrix( f, R, t ):

    K = np.array( [ [ f, 0, cx ], [ 0, f, cy ], [ 0, 0, 1 ] ] )
    return K @ np.hstack( [ R, t.reshape( 3, 1 ) ] )


def triangulate_multiview( obs_list, f, rvecs, tvecs ):

    A = []

    for ci, u, v in obs_list:

        R, _ = cv2.Rodrigues( rvecs[ ci ] )
        P = projection_matrix( f[ ci ], R, tvecs[ ci ] )

        A.append( u * P[ 2 ] - P[ 0 ] )
        A.append( v * P[ 2 ] - P[ 1 ] )

    _, _, Vt = np.linalg.svd( np.array( A ) )
    Xh = Vt[ -1 ]

    if abs( Xh[ 3 ] ) < 1e-12:
        return None

    return Xh[ : 3 ] / Xh[ 3 ]


def build_points( wand_rvecs, wand_tvecs, mp_X, n_wand_pose ):

    n_mp = len( mp_X )
    X = np.zeros( ( n_wand_pose * n_wand_k + n_mp, 3 ) )

    for pi in range( n_wand_pose ):

        R_w, _ = cv2.Rodrigues( wand_rvecs[ pi ] )

        for k in range( n_wand_k ):
            X[ pi * n_wand_k + k ] = R_w @ wand_local[ k ] + wand_tvecs[ pi ]

    if n_mp > 0:
        X[ n_wand_pose * n_wand_k : ] = mp_X.reshape( -1, 3 )

    return X


_other_cams = [ ci for ci in range( n_cam ) if ci != ref_cam ]


def pack( f, rvecs, tvecs, wand_rvecs, wand_tvecs, mp_X ):

    return np.concatenate(
        [
            f,
            rvecs[ _other_cams ].ravel(),
            tvecs[ _other_cams ].ravel(),
            wand_rvecs.ravel(),
            wand_tvecs.ravel(),
            mp_X.ravel(),
        ]
    )


def make_unpack( n_wand_pose, n_wand_point, n_point ):

    n_mp = n_point - n_wand_point

    def unpack( params ):

        i = 0
        f = params[ i : i + n_cam ]
        i += n_cam

        rvecs = np.zeros( ( n_cam, 3 ) )
        rvecs[ _other_cams ] = params[ i : i + ( n_cam - 1 ) * 3 ].reshape( ( n_cam - 1, 3 ) )
        i += ( n_cam - 1 ) * 3

        tvecs = np.zeros( ( n_cam, 3 ) )
        tvecs[ _other_cams ] = params[ i : i + ( n_cam - 1 ) * 3 ].reshape( ( n_cam - 1, 3 ) )
        i += ( n_cam - 1 ) * 3

        wand_rvecs = params[ i : i + n_wand_pose * 3 ].reshape( ( n_wand_pose, 3 ) )
        i += n_wand_pose * 3

        wand_tvecs = params[ i : i + n_wand_pose * 3 ].reshape( ( n_wand_pose, 3 ) )
        i += n_wand_pose * 3

        mp_X = params[ i : ].reshape( ( n_mp, 3 ) ) if n_mp > 0 else np.zeros( ( 0, 3 ) )
        X = build_points( wand_rvecs, wand_tvecs, mp_X, n_wand_pose )

        return f, rvecs, tvecs, wand_rvecs, wand_tvecs, mp_X, X

    return unpack


def make_residuals( unpack, rvecs0_ref, tvecs0_ref ):

    def residuals( params, observations ):

        f, rvecs, tvecs, _, _, _, X = unpack( params )

        R_list = [ cv2.Rodrigues( rvecs[ ci ] )[ 0 ] for ci in range( n_cam ) ]

        res = []

        for cam_idx, pt_idx, u_obs, v_obs, weight in observations:

            Xc = R_list[ cam_idx ] @ X[ pt_idx ] + tvecs[ cam_idx ]

            if Xc[ 2 ] <= 1e-3:
                res.extend( [ 1e3 * np.sqrt( weight ), 1e3 * np.sqrt( weight ) ] )
                continue

            u_proj = f[ cam_idx ] * Xc[ 0 ] / Xc[ 2 ] + cx
            v_proj = f[ cam_idx ] * Xc[ 1 ] / Xc[ 2 ] + cy
            w = np.sqrt( weight )

            res.append( ( u_proj - u_obs ) * w )
            res.append( ( v_proj - v_obs ) * w )

        for ci in _other_cams:
            res.extend( ( rvecs[ ci ] - rvecs0_ref[ ci ] ) * cam_reg_weight )
            res.extend( ( tvecs[ ci ] - tvecs0_ref[ ci ] ) * cam_reg_weight )

        return np.array( res )

    return residuals


def make_jac_sparsity( n_wand_pose, n_wand_point, n_point, observations ):
    """
    各残差がどのパラメータに依存するかを示すスパース行列を返す。
    これを least_squares に渡すことで有限差分の計算量を大幅削減できる。
    """

    n_mp    = n_point - n_wand_point
    n_obs   = len( observations )
    n_res   = 2 * n_obs + 6 * ( n_cam - 1 )
    n_param = n_cam + 2 * ( n_cam - 1 ) * 3 + 2 * n_wand_pose * 3 + n_mp * 3

    # パラメータブロックの先頭インデックス
    p_f         = 0
    p_rvec      = n_cam
    p_tvec      = n_cam + ( n_cam - 1 ) * 3
    p_wand_rvec = n_cam + 2 * ( n_cam - 1 ) * 3
    p_wand_tvec = p_wand_rvec + n_wand_pose * 3
    p_mp        = p_wand_tvec + n_wand_pose * 3

    S = lil_matrix( ( n_res, n_param ), dtype = np.int8 )

    for obs_i, ( cam_idx, pt_idx, *_ ) in enumerate( observations ):

        r0, r1 = 2 * obs_i, 2 * obs_i + 1

        # 焦点距離
        S[ r0, p_f + cam_idx ] = 1
        S[ r1, p_f + cam_idx ] = 1

        # カメラ外部パラメータ（ref_cam は固定なのでパラメータに含まれない）
        if cam_idx != ref_cam:
            loc = cam_idx if cam_idx < ref_cam else cam_idx - 1
            for d in range( 3 ):
                S[ r0, p_rvec + loc * 3 + d ] = 1
                S[ r1, p_rvec + loc * 3 + d ] = 1
                S[ r0, p_tvec + loc * 3 + d ] = 1
                S[ r1, p_tvec + loc * 3 + d ] = 1

        # 3D点
        if pt_idx < n_wand_point:
            pose_i = pt_idx // n_wand_k
            for d in range( 3 ):
                S[ r0, p_wand_rvec + pose_i * 3 + d ] = 1
                S[ r1, p_wand_rvec + pose_i * 3 + d ] = 1
                S[ r0, p_wand_tvec + pose_i * 3 + d ] = 1
                S[ r1, p_wand_tvec + pose_i * 3 + d ] = 1
        else:
            local_i = pt_idx - n_wand_point
            for d in range( 3 ):
                S[ r0, p_mp + local_i * 3 + d ] = 1
                S[ r1, p_mp + local_i * 3 + d ] = 1

    # カメラ正則化残差
    for local_i, ci in enumerate( _other_cams ):
        base = 2 * n_obs + local_i * 6
        for d in range( 3 ):
            S[ base + d,     p_rvec + local_i * 3 + d ] = 1
            S[ base + 3 + d, p_tvec + local_i * 3 + d ] = 1

    return S.tocsr()


def reprojection_error_per_camera( observations, f, rvecs, tvecs, X ):
    """各カメラの再投影誤差を集計して rich テーブルで表示し、JSON にも保存する。"""

    import json
    from collections import defaultdict
    errors = defaultdict( list )

    for cam_idx, pt_idx, u_obs, v_obs, weight in observations:
        R, _ = cv2.Rodrigues( rvecs[ cam_idx ] )
        Xc   = R @ X[ pt_idx ] + tvecs[ cam_idx ]

        if Xc[ 2 ] <= 1e-3:
            continue

        u_proj = f[ cam_idx ] * Xc[ 0 ] / Xc[ 2 ] + cx
        v_proj = f[ cam_idx ] * Xc[ 1 ] / Xc[ 2 ] + cy

        err = np.sqrt( ( u_proj - u_obs ) ** 2 + ( v_proj - v_obs ) ** 2 )
        errors[ cam_idx ].append( err )

    tbl = Table( title = 'Reprojection Error per Camera [px]', border_style = 'dim' )
    tbl.add_column( 'Camera', style = 'bold cyan' )
    tbl.add_column( 'N obs',  justify = 'right' )
    tbl.add_column( 'Mean',   justify = 'right', style = 'yellow' )
    tbl.add_column( 'Median', justify = 'right' )
    tbl.add_column( 'RMS',    justify = 'right', style = 'magenta' )
    tbl.add_column( 'Max',    justify = 'right', style = 'red' )

    summary = {}
    for ci in range( n_cam ):
        errs = np.array( errors[ ci ] )
        if len( errs ) == 0:
            tbl.add_row( camera_names[ ci ], '0', '-', '-', '-', '-' )
            summary[ camera_names[ ci ] ] = { 'n_obs': 0 }
        else:
            tbl.add_row(
                camera_names[ ci ],
                str( len( errs ) ),
                f'{errs.mean():.2f}',
                f'{np.median( errs ):.2f}',
                f'{np.sqrt( ( errs**2 ).mean() ):.2f}',
                f'{errs.max():.2f}',
            )
            summary[ camera_names[ ci ] ] = {
                'n_obs':  int( len( errs ) ),
                'mean':   float( errs.mean() ),
                'median': float( np.median( errs ) ),
                'rms':    float( np.sqrt( ( errs**2 ).mean() ) ),
                'max':    float( errs.max() ),
            }

    console.print( tbl )
    console.print()

    # sync_config.json に reprojection_errors を追記（なければ新規作成）
    from pathlib import Path as _Path
    _cfg_path = _Path( 'sync_config.json' )
    try:
        _cfg_data = json.loads( _cfg_path.read_text( encoding='utf-8' ) ) if _cfg_path.exists() else {}
    except Exception:
        _cfg_data = {}
    _cfg_data[ 'reprojection_errors' ] = summary
    _cfg_path.write_text( json.dumps( _cfg_data, indent=2, ensure_ascii=False ), encoding='utf-8' )
    console.print( '[bold green]✓[/bold green] Saved [cyan]reprojection_errors[/cyan] → [cyan]sync_config.json[/cyan]' )
    console.print()


def triangulate_all_frames( mp_dataframes, landmark_names, f, rvecs, tvecs, max_frames = None ):

    records = []
    _tri_src   = next( ( df for df in mp_dataframes if len( df ) > 0 ), mp_dataframes[ 0 ] )
    all_frames = _tri_src[ 'frame' ].values

    if max_frames is not None and len( all_frames ) > max_frames:
        all_frames = all_frames[ : max_frames ]

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

                for ci in range( n_cam ):

                    row = mp_dataframes[ ci ][ mp_dataframes[ ci ][ 'frame' ] == frame ]

                    if len( row ) == 0:
                        continue

                    vis = row[ f'{landmark}_v' ].values[ 0 ]

                    if pd.isna( vis ) or vis < v_thresh:
                        continue

                    u_px = row[ f'{landmark}_x' ].values[ 0 ] * image_width
                    v_px = row[ f'{landmark}_y' ].values[ 0 ] * image_height

                    obs.append( ( ci, u_px, v_px, vis ) )

                if len( obs ) < 2:
                    continue

                X = triangulate_multiview( [ ( ci, u, v ) for ci, u, v, _ in obs ], f, rvecs, tvecs )

                if X is None:
                    continue

                records.append(
                    {
                        'frame': int( frame ),
                        'landmark': landmark,
                        'x': X[ 0 ],
                        'y': X[ 1 ],
                        'z': X[ 2 ],
                        'n_views': len( obs ),
                        'mean_visibility': float( np.mean( [ v for *_, v in obs ] ) ),
                    }
                )

            progress.advance( task )

    if not records:
        return pd.DataFrame( columns = [ 'frame', 'landmark', 'x', 'y', 'z', 'n_views', 'mean_visibility' ] )
    return pd.DataFrame( records )


# --------------------------------------------------
# Visualization
# --------------------------------------------------

def plot_camera_poses( cam_positions, names, output_path = 'camera_poses.png' ):

    positions = np.array( cam_positions )
    colors    = plt.cm.tab10( np.linspace( 0, 1, len( names ) ) )

    fig, axes = plt.subplots( 1, 3, figsize = ( 15, 5 ) )
    fig.suptitle( 'Camera Positions', fontsize = 14, fontweight = 'bold' )

    planes = [
        ( 0, 1, 'X', 'Y', 'XY plane (top view)' ),
        ( 0, 2, 'X', 'Z', 'XZ plane (front view)' ),
        ( 1, 2, 'Y', 'Z', 'YZ plane (side view)' ),
    ]

    for ax, ( xi, yi, xl, yl, title ) in zip( axes, planes ):

        for i, ( pos, name ) in enumerate( zip( positions, names ) ):
            ax.scatter( pos[ xi ], pos[ yi ], color = colors[ i ], s = 80, zorder = 3 )
            ax.annotate(
                name,
                ( pos[ xi ], pos[ yi ] ),
                textcoords = 'offset points',
                xytext = ( 6, 4 ),
                fontsize = 8,
                color = colors[ i ],
            )

        ax.scatter( 0, 0, marker = '+', s = 120, color = 'black', zorder = 4, linewidths = 2 )
        ax.set_xlabel( xl )
        ax.set_ylabel( yl )
        ax.set_title( title )
        ax.set_aspect( 'equal' )
        ax.grid( True, linestyle = '--', alpha = 0.4 )

    legend_patches = [
        mpatches.Patch( color = colors[ i ], label = names[ i ] )
        for i in range( len( names ) )
    ]
    fig.legend( handles = legend_patches, loc = 'lower center', ncol = len( names ), fontsize = 9 )

    plt.tight_layout( rect = [ 0, 0.06, 1, 1 ] )
    plt.savefig( output_path, dpi = 150, bbox_inches = 'tight' )
    plt.close()

    console.print(
        f'[bold green]✓[/bold green] Saved camera pose plot [cyan]{output_path}[/cyan]'
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # sync_config.json からキャリブレーション設定・CSV パスを読み込む
    import json as _json
    from pathlib import Path as _Path
    _cfg_file = _Path( 'sync_config.json' )
    _cfg_all: dict = {}
    if _cfg_file.exists():
        try:
            _cfg_all = _json.loads( _cfg_file.read_text( encoding='utf-8' ) )
        except Exception:
            pass

    _calib_cfg          = _cfg_all.get( 'calib', {} )
    _pose3d_cfg         = _cfg_all.get( 'pose3d', {} )

    _wand_csv_path      = _calib_cfg.get( 'wand_csv', 'wand_annotations.csv' )
    # 絶対パスが存在しない場合（Colab 等の別環境）→ ファイル名のみで再検索
    if _wand_csv_path and not Path( _wand_csv_path ).exists():
        _fallback = Path( _wand_csv_path ).name
        if Path( _fallback ).exists():
            console.print( f'[dim]Wand CSV path not found, using: {_fallback}[/dim]' )
            _wand_csv_path = _fallback
    _start_frames_cfg   = _calib_cfg.get( 'start_frames_per_cam', {} )
    _max_calib_frames   = _calib_cfg.get( 'max_calib_frames', None )

    # calib.landmarks を優先し、なければ pose3d.landmarks を使用
    _calib_lm    = _calib_cfg.get( 'landmarks', {} )
    _pose3d_lm   = _pose3d_cfg.get( 'landmarks', {} )
    _landmarks_store = _calib_lm if _calib_lm else _pose3d_lm

    console.print( f'[dim]Wand CSV      : {_wand_csv_path}[/dim]' )
    console.print( f'[dim]Landmark src  : {"calib.landmarks" if _calib_lm else "pose3d.landmarks"}[/dim]' )
    console.print( f'[dim]Start frames  : {_start_frames_cfg}[/dim]' )
    console.print( f'[dim]Max BA frames : {_max_calib_frames}[/dim]' )
    console.print()

    # (cam_idx, pt_idx, u, v, weight)
    observations = []
    point_index  = 0

    from pathlib import Path as _WandPath
    _wand_csv = _WandPath( _wand_csv_path )
    if _wand_csv.exists():
        wand_df      = pd.read_csv( _wand_csv_path, encoding='utf-8' )
        pose_names   = sorted( wand_df[ 'pose' ].unique() )
        point_labels = [ '0.0m', '0.5m', '1.0m', '1.5m' ]
        n_wand_pose  = len( pose_names )

        for pose_i, pose in enumerate( pose_names ):
            for k, label in enumerate( point_labels ):
                for ci in range( n_cam ):
                    cam = camera_names[ ci ]
                    row = wand_df[
                        ( wand_df[ 'pose' ] == pose )
                        & ( wand_df[ 'point_label' ] == label )
                        & ( wand_df[ 'camera' ] == cam )
                    ]
                    u, v = row[ [ 'u', 'v' ] ].values[ 0 ]
                    observations.append( ( ci, point_index, u, v, 1.0 ) )
                point_index += 1
    else:
        console.print(
            f'[yellow]Wand CSV not found ({_wand_csv_path}) '
            f'— MediaPipe のみでキャリブレーションを実行します[/yellow]'
        )
        console.print()
        pose_names  = []
        n_wand_pose = 0

    n_wand_point = point_index

    # sync_config.json からランドマークを読み込む（ないカメラは空 DataFrame でスキップ）
    mp_dataframes = []
    for name in camera_names:
        lm_entry = _landmarks_store.get( name )
        if lm_entry:
            df = pd.DataFrame( lm_entry[ 'data' ], columns = lm_entry[ 'columns' ] )
            mp_dataframes.append( df )
            console.print( f'[dim]{name}: loaded from sync_config.json ({len(df)} rows)[/dim]' )
        else:
            console.print( f'[yellow]{name}: ランドマークなし — スキップ（観測データなし）[/yellow]' )
            mp_dataframes.append( pd.DataFrame( columns=['frame'] ) )

    # start_frames_per_cam フィルタ: キャリブレーション開始フレーム以降のみ使用
    for ci, name in enumerate( camera_names ):
        sf = _start_frames_cfg.get( name )
        if sf is not None and len( mp_dataframes[ ci ] ) > 0:
            mp_dataframes[ ci ] = (
                mp_dataframes[ ci ][ mp_dataframes[ ci ][ 'frame' ] >= sf ]
                .reset_index( drop=True )
            )
            console.print( f'[dim]{name}: frames >= {sf}  ({len(mp_dataframes[ci])} rows)[/dim]' )

    # landmark_names を最初の有効な DataFrame から取得
    _lm_src = next(
        ( df for df in mp_dataframes if len( df.columns ) > 1 ), mp_dataframes[ 0 ]
    )
    landmark_names = [
        col[ : -2 ] for col in _lm_src.columns if col.endswith( '_v' )
    ]

    # BA に使うフレームをランダムサンプリング（最初の有効 DataFrame を基準に）
    _first_valid = next(
        ( df for df in mp_dataframes if len( df ) > 0 ), mp_dataframes[ 0 ]
    )
    all_frames = _first_valid[ 'frame' ].values
    rng = np.random.default_rng( 42 )

    # 人が検出されているフレームのみ使用（いずれかのカメラで v_thresh 以上のランドマークがある）
    _detected_frame_set = set()
    for _df in mp_dataframes:
        if len( _df ) == 0:
            continue
        _v_cols = [ c for c in _df.columns if c.endswith( '_v' ) ]
        if not _v_cols:
            continue
        _mask = ( _df[ _v_cols ] >= v_thresh ).any( axis=1 )
        _detected_frame_set.update( _df.loc[ _mask, 'frame' ].tolist() )
    if _detected_frame_set:
        _n_before = len( all_frames )
        all_frames = np.array( sorted( _detected_frame_set ) )
        console.print(
            f'[dim]人検出フレーム (v≥{v_thresh}): {len(all_frames)} / {_n_before}[/dim]'
        )
    else:
        console.print( f'[red]有効な検出フレームがありません (v_thresh={v_thresh})。[/red]' )
        return

    if _max_calib_frames is not None and len( all_frames ) > _max_calib_frames:
        frame_list = rng.choice( all_frames, size=_max_calib_frames, replace=False )
        frame_list = np.sort( frame_list )
        console.print( f'[dim]BA frame list: random {_max_calib_frames} / {len(all_frames)} frames[/dim]' )
    else:
        frame_list = all_frames
        console.print( f'[dim]BA frame list: all {len(frame_list)} frames[/dim]' )
    console.print()

    for frame in frame_list:

        for landmark in landmark_names:

            visible_cams = []

            for ci in range( n_cam ):

                row = mp_dataframes[ ci ][ mp_dataframes[ ci ][ 'frame' ] == frame ]

                if len( row ) == 0:
                    continue

                vis = row[ f'{landmark}_v' ].values[ 0 ]

                if pd.isna( vis ) or vis < v_thresh:
                    continue

                u_px = row[ f'{landmark}_x' ].values[ 0 ] * image_width
                v_px = row[ f'{landmark}_y' ].values[ 0 ] * image_height

                visible_cams.append( ( ci, u_px, v_px, vis ) )

            if len( visible_cams ) >= 2:

                for ci, u_px, v_px, vis in visible_cams:
                    observations.append( ( ci, point_index, u_px, v_px, vis ) )

                point_index += 1

    n_point = point_index
    n_mp    = n_point - n_wand_point

    console.print(
        Panel(
            Text( 'Camera Calibration via Bundle Adjustment', style = 'bold cyan', justify = 'center' ),
            border_style = 'cyan',
            padding = ( 0, 2 ),
        )
    )

    data_table = Table( title = 'Dataset', show_header = False, border_style = 'dim' )
    data_table.add_column( 'Key',   style = 'bold green' )
    data_table.add_column( 'Value', style = 'white' )
    data_table.add_row( 'Wand poses',          f'{n_wand_pose}' )
    data_table.add_row( 'Wand obs points',     f'{n_wand_point}' )
    data_table.add_row( 'MediaPipe points',    f'{n_mp}' )
    data_table.add_row( 'Total observations',  f'{len( observations ):,}' )
    console.print( data_table )
    console.print()

    # --------------------------------------------------
    # Init cameras
    # --------------------------------------------------

    f_init = 800.0
    f0     = np.full( n_cam, f_init )

    rvecs0 = np.zeros( ( n_cam, 3 ) )
    tvecs0 = np.zeros( ( n_cam, 3 ) )

    # 右手座標系: ピッチャー = 原点
    #   +Y = ホーム方向, -Y = 外野方向
    #   +Z = 上（鉛直）
    #   +X = 3塁側（X × Y = Z を満たす方向）, -X = 1塁側
    cam_init_positions = np.array( [
        [  42.0, -42.0,  2.0 ],  # cam1: レフト（3塁側外野）  (~60m)
        [  20.0,  25.0,  3.0 ],  # cam2: 3塁側ベンチ上        (~32m)
        [  15.0,  47.0,  2.0 ],  # cam3: バックネット裏3塁側  (~50m)
        [  -5.0,  50.0,  2.0 ],  # cam4: ほぼ正面・1塁側      (~50m)
        [ -20.0,  25.0,  2.0 ],  # cam5: 1塁ベンチ            (~32m)
        [ -42.0, -42.0,  2.0 ],  # cam6: ライト（1塁側外野）  (~60m)
    ] )

    for ci in range( n_cam ):
        rvecs0[ ci ], tvecs0[ ci ] = look_at_rvec_tvec(
            cam_init_positions[ ci ], up = np.array( [ 0.0, 0.0, 1.0 ] )
        )

    # ref_cam のカメラ座標系を最適化ワールドとして使うため
    # 全カメラのrvec/tvecをref_cam基準（最適化世界）に変換する
    # 変換後: rvecs0[ref_cam] = 0, tvecs0[ref_cam] = 0
    R_ref, _ = cv2.Rodrigues( rvecs0[ ref_cam ] )
    t_ref     = tvecs0[ ref_cam ].copy()

    # ピッチャーの位置（フィールド原点）は変換前の ref_cam の tvec
    pitcher_in_opt_world = t_ref.copy()

    for ci in range( n_cam ):
        R_ci, _ = cv2.Rodrigues( rvecs0[ ci ] )
        t_ci    = tvecs0[ ci ].copy()
        R_ci_opt = R_ci @ R_ref.T
        t_ci_opt = t_ci - R_ci_opt @ t_ref
        rvec_opt, _ = cv2.Rodrigues( R_ci_opt )
        rvecs0[ ci ] = rvec_opt.ravel()
        tvecs0[ ci ] = t_ci_opt

    rvecs0_ref = rvecs0.copy()
    tvecs0_ref = tvecs0.copy()

    # ワンドの初期位置: ピッチャー付近（ref_cam から見た先）に 1m 間隔で展開
    wand_rvecs0 = np.zeros( ( n_wand_pose, 3 ) )
    wand_tvecs0 = np.zeros( ( n_wand_pose, 3 ) )

    right_dir = np.array( [ 1.0, 0.0, 0.0 ] )  # 最適化世界の X 軸（ref_cam の右方向）

    for pi in range( n_wand_pose ):
        wand_tvecs0[ pi ] = pitcher_in_opt_world + right_dir * float( pi )

    obs_by_point = {}

    for cam_idx, pt_idx, u, v, _ in observations:
        obs_by_point.setdefault( pt_idx, [] ).append( ( cam_idx, u, v ) )

    mp_X0 = np.zeros( ( n_mp, 3 ) )

    for local_i, pt_idx in enumerate( range( n_wand_point, n_point ) ):

        X = triangulate_multiview( obs_by_point[ pt_idx ], f0, rvecs0, tvecs0 )

        if X is None:
            X = np.array( [ 0.0, 1.0, 1.5 ] )

        mp_X0[ local_i ] = X

    unpack   = make_unpack( n_wand_pose, n_wand_point, n_point )
    residual = make_residuals( unpack, rvecs0_ref, tvecs0_ref )

    params0 = pack( f0, rvecs0, tvecs0, wand_rvecs0, wand_tvecs0, mp_X0 )

    r0 = residual( params0, observations )

    if not np.all( np.isfinite( r0 ) ):
        raise ValueError( f'Initial residuals not finite: {np.sum( ~np.isfinite( r0 ) )} bad values' )

    console.rule( '[bold yellow]Bundle Adjustment[/bold yellow]' )
    console.print( f'[bold]Optimizing [cyan]{len( params0 )}[/cyan] parameters...[/bold]' )
    console.print()

    sparsity = make_jac_sparsity( n_wand_pose, n_wand_point, n_point, observations )

    # パラメータブロックごとに代表スケールを指定（単位が px / rad / m とバラバラなため）
    x_scale = np.concatenate( [
        np.full( n_cam,          500.0 ),   # 焦点距離 (~800 px)
        np.full( (n_cam-1)*3,    0.5   ),   # カメラ回転 (~1 rad)
        np.full( (n_cam-1)*3,    30.0  ),   # カメラ位置 (~50 m)
        np.full( n_wand_pose*3,  0.5   ),   # ワンド回転
        np.full( n_wand_pose*3,  1.0   ),   # ワンド位置 (~1 m)
        np.full( n_mp*3,         3.0   ),   # MediaPipe 3D点 (~数 m)
    ] )

    # 焦点距離の崩壊（f→0）を防ぐ bounds
    n_params = len( params0 )
    lb = np.full( n_params, -np.inf )
    ub = np.full( n_params,  np.inf )
    lb[ : n_cam ] = 300.0   # 最小焦点距離 (px)
    ub[ : n_cam ] = 2000.0  # 最大焦点距離 (px)

    common_kwargs = dict(
        args         = ( observations, ),
        method       = 'trf',
        jac_sparsity = sparsity,
        x_scale      = x_scale,
        bounds       = ( lb, ub ),
    )

    console.print( '[bold cyan]Phase 1[/bold cyan]  loss=linear  (粗い収束)' )
    result1 = least_squares(
        residual,
        params0,
        loss     = 'linear',
        max_nfev = 500,
        verbose  = 2,
        **common_kwargs,
    )
    console.print()

    console.print( '[bold cyan]Phase 2[/bold cyan]  loss=huber  (精密化・外れ値除去)' )
    result = least_squares(
        residual,
        result1.x,
        loss     = 'huber',
        f_scale  = huber_f_scale,
        max_nfev = 500,
        verbose  = 2,
        **common_kwargs,
    )

    f, rvecs, tvecs, wand_rvecs, wand_tvecs, mp_X, X = unpack( result.x )

    console.print()
    console.rule( '[bold green]Results[/bold green]' )

    opt_table = Table( title = 'Optimization', show_header = False, border_style = 'dim' )
    opt_table.add_column( 'Key',   style = 'bold magenta' )
    opt_table.add_column( 'Value', style = 'white' )
    opt_table.add_row( 'Cost',   f'{result.cost:.4f}' )
    opt_table.add_row( 'NFev',   f'{result.nfev}' )
    opt_table.add_row( 'Status', result.message )
    console.print( opt_table )
    console.print()

    reprojection_error_per_camera( observations, f, rvecs, tvecs, X )

    cam_table = Table( title = 'Camera Intrinsics & Positions (World)', border_style = 'dim' )
    cam_table.add_column( 'Camera', style = 'bold cyan' )
    cam_table.add_column( 'Focal Length', style = 'yellow', justify = 'right' )
    cam_table.add_column( 'X',   style = 'white', justify = 'right' )
    cam_table.add_column( 'Y',   style = 'white', justify = 'right' )
    cam_table.add_column( 'Z',   style = 'white', justify = 'right' )

    cam_positions = []

    for ci in range( n_cam ):

        R, _ = cv2.Rodrigues( rvecs[ ci ] )
        cam_pos = -R.T @ tvecs[ ci ]
        cam_positions.append( cam_pos )
        cam_table.add_row(
            camera_names[ ci ],
            f'{f[ ci ]:.1f}',
            f'{cam_pos[ 0 ]:.3f}',
            f'{cam_pos[ 1 ]:.3f}',
            f'{cam_pos[ 2 ]:.3f}',
        )

    console.print( cam_table )
    console.print()

    if n_wand_pose > 0:
        wand_table = Table( title = 'Wand Poses (Translation)', border_style = 'dim' )
        wand_table.add_column( 'Pose', style = 'bold cyan' )
        wand_table.add_column( 'tx', style = 'white', justify = 'right' )
        wand_table.add_column( 'ty', style = 'white', justify = 'right' )
        wand_table.add_column( 'tz', style = 'white', justify = 'right' )
        for pi, pose in enumerate( pose_names ):
            t = wand_tvecs[ pi ]
            wand_table.add_row( pose, f'{t[0]:.3f}', f'{t[1]:.3f}', f'{t[2]:.3f}' )
        console.print( wand_table )
        console.print()

    calib_path = 'camera_calibration.npz'
    np.savez(
        calib_path,
        camera_names = camera_names,
        f = f,
        rvecs = rvecs,
        tvecs = tvecs,
        image_width = image_width,
        image_height = image_height,
    )
    console.print(
        f'[bold green]✓[/bold green] Saved calibration [cyan]{calib_path}[/cyan]'
    )
    # sync_config.json に NPZ パスを保存
    try:
        from pathlib import Path as _Path2
        _cfg_p2 = _Path2('sync_config.json')
        _cfg_d2 = json.loads(_cfg_p2.read_text(encoding='utf-8')) if _cfg_p2.exists() else {}
        if 'calib' not in _cfg_d2:
            _cfg_d2['calib'] = {}
        _cfg_d2['calib']['npz_path'] = str(_Path2(calib_path).resolve())
        _cfg_p2.write_text(json.dumps(_cfg_d2, indent=2, ensure_ascii=False), encoding='utf-8')
    except Exception:
        pass
    console.print()

    console.rule( '[bold yellow]Triangulation[/bold yellow]' )
    total_frames = len( next( ( df for df in mp_dataframes if len( df ) > 0 ), mp_dataframes[ 0 ] ) )
    limit_note = f'{max_tri_frames:,} / {total_frames:,}' if max_tri_frames and total_frames > max_tri_frames else f'{total_frames:,} (all)'
    console.print( f'Frames to triangulate: [cyan]{limit_note}[/cyan]' )
    console.print()
    df_3d = triangulate_all_frames( mp_dataframes, landmark_names, f, rvecs, tvecs, max_frames = max_tri_frames )

    output_csv = 'landmarks_3d.csv'

    if len( df_3d ) == 0:
        console.print( '[yellow]Warning: 三角測量結果が空です。start_frames_per_cam または MediaPipe CSV を確認してください。[/yellow]' )
    else:
        df_3d.to_csv( output_csv, index = False, encoding='utf-8' )

        tri_table = Table( title = 'Triangulation Result', show_header = False, border_style = 'dim' )
        tri_table.add_column( 'Key',   style = 'bold green' )
        tri_table.add_column( 'Value', style = 'white' )
        tri_table.add_row( 'Output',    output_csv )
        tri_table.add_row( 'Rows',      f'{len( df_3d ):,}' )
        tri_table.add_row( 'Frames',    f'{df_3d["frame"].nunique():,}' )
        tri_table.add_row( 'Landmarks', f'{df_3d["landmark"].nunique()}' )
        console.print()
        console.print( tri_table )
        console.print()
        console.print(
            f'[bold green]✓[/bold green] Saved [cyan]{output_csv}[/cyan]'
        )

    plot_camera_poses( cam_positions, camera_names )


if __name__ == '__main__':
    main()
