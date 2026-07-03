import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io as _io
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path

import torch
from ultralytics import YOLO

from rich.console import Console
from rich.panel   import Panel
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

# ultralytics が自動的にダウンロード・キャッシュする
DEFAULT_MODEL = 'yolo11x-pose.pt'

# 人物検出の信頼度閾値
conf_threshold = 0.5
# キーポイント信頼度がこれ未満なら NaN として保存
kp_threshold   = 0.3

# YOLO Pose (COCO 17 keypoints) → MediaPipe 互換の列名
# recog_mediapipe.py と共通のランドマーク名を使うことで下流処理を共用できる
LANDMARK_NAMES = [
    'NOSE',           # 0
    'LEFT_EYE',       # 1
    'RIGHT_EYE',      # 2
    'LEFT_EAR',       # 3
    'RIGHT_EAR',      # 4
    'LEFT_SHOULDER',  # 5
    'RIGHT_SHOULDER', # 6
    'LEFT_ELBOW',     # 7
    'RIGHT_ELBOW',    # 8
    'LEFT_WRIST',     # 9
    'RIGHT_WRIST',    # 10
    'LEFT_HIP',       # 11
    'RIGHT_HIP',      # 12
    'LEFT_KNEE',      # 13
    'RIGHT_KNEE',     # 14
    'LEFT_ANKLE',     # 15
    'RIGHT_ANKLE',    # 16
]


def _get_device() -> str:
    """利用可能なアクセラレータを自動選択する (CUDA > MPS > CPU)。"""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _extract_keypoints(result, width: int, height: int) -> dict | None:
    """
    ultralytics の Results オブジェクトから最高信頼度の人物の
    キーポイントを抽出し、正規化座標に変換して返す。
    人物未検出なら None を返す。
    """
    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    # 複数人がいる場合は bbox の信頼度が最高の人物を採用
    if result.boxes is not None and len(result.boxes.conf) > 0:
        best = int(result.boxes.conf.argmax())
    else:
        best = 0

    # xyn: 正規化済み座標 [N, 17, 2], conf: [N, 17]
    xyn  = result.keypoints.xyn[best]   # [17, 2]  (x_norm, y_norm)
    conf = result.keypoints.conf[best]  # [17]

    kps = {}
    for i, name in enumerate(LANDMARK_NAMES):
        x_n = float(xyn[i, 0])
        y_n = float(xyn[i, 1])
        c   = float(conf[i])
        # 信頼度が閾値未満のキーポイントは NaN（MediaPipe の visibility=0 相当）
        if c < kp_threshold:
            x_n = y_n = np.nan
        kps[name] = (x_n, y_n, c)
    return kps


def _build_row(frame_idx: int, kps: dict | None) -> dict:
    """フレームインデックスとキーポイント辞書から CSV 行を組み立てる。"""
    row = {'frame': frame_idx}
    if kps is not None:
        for name, (x, y, c) in kps.items():
            row[f'{name}_x'] = x
            row[f'{name}_y'] = y
            row[f'{name}_z'] = 0.0   # YOLO Pose は奥行きを持たない
            row[f'{name}_v'] = c
    else:
        for name in LANDMARK_NAMES:
            row[f'{name}_x'] = np.nan
            row[f'{name}_y'] = np.nan
            row[f'{name}_z'] = np.nan
            row[f'{name}_v'] = np.nan
    return row


# ------------------------------------------------------------------
# スタンドアロン処理（単一動画ファイル）
# ------------------------------------------------------------------

def pose_recog(folder_path: str, video_name: str, model_name: str = DEFAULT_MODEL):

    video_path = os.path.join(folder_path, f'{video_name}.mp4')
    output_csv = f'{video_name}.csv'

    console = Console(legacy_windows=False)
    device  = _get_device()

    console.print(
        Panel(
            Text('YOLO Pose Landmark Extraction', style='bold cyan', justify='center'),
            border_style='cyan',
            padding=(0, 2),
        )
    )

    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration  = frame_num / fps if fps > 0 else 0

    info_table = Table(title='Video Info', show_header=False, border_style='dim')
    info_table.add_column('Key',   style='bold green')
    info_table.add_column('Value', style='white')
    info_table.add_row('Path',            str(video_path))
    info_table.add_row('Resolution',      f'{width} x {height}')
    info_table.add_row('FPS',             f'{fps:.2f}')
    info_table.add_row('Frames',          f'{frame_num:,}')
    info_table.add_row('Duration',        f'{duration:.1f} sec')
    info_table.add_row('Model',           model_name)
    info_table.add_row('Device',          device)
    info_table.add_row('Conf threshold',  f'{conf_threshold:.2f}')
    info_table.add_row('KP threshold',    f'{kp_threshold:.2f}')
    console.print(info_table)
    console.print()

    records       = []
    frame_idx     = 0
    detected_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn('[bold blue]{task.description}'),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn('*'),
        TimeElapsedColumn(),
        TextColumn('*'),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:

        task = progress.add_task('Processing frames', total=frame_num or None)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, device=device, conf=conf_threshold,
                                    verbose=False, stream=False)
            kps = _extract_keypoints(results[0], width, height)
            if kps is not None:
                detected_count += 1

            records.append(_build_row(frame_idx, kps))
            progress.advance(task)
            frame_idx += 1

    cap.release()

    df = pd.DataFrame(records)
    detection_rate = (detected_count / len(df) * 100) if len(df) else 0

    summary_table = Table(title='Summary', show_header=False, border_style='dim')
    summary_table.add_column('Key',   style='bold magenta')
    summary_table.add_column('Value', style='white')
    summary_table.add_row('Processed frames', f'{len(df):,}')
    summary_table.add_row('Pose detected',    f'{detected_count:,}')
    summary_table.add_row('Detection rate',   f'{detection_rate:.1f}%')
    summary_table.add_row('Columns',          f'{len(df.columns):,}')
    summary_table.add_row('Output',           output_csv)
    console.print()
    console.print(summary_table)
    console.print()

    preview_cols = ['frame', 'NOSE_x', 'NOSE_y', 'NOSE_z', 'NOSE_v',
                    'LEFT_WRIST_x', 'RIGHT_WRIST_x']
    preview_cols = [c for c in preview_cols if c in df.columns]
    console.print('[bold yellow]Preview[/bold yellow]')
    console.print(df[preview_cols].head())

    df.to_csv(output_csv, index=False)
    console.print()
    console.print(
        f'[bold green]+[/bold green] Saved [cyan]{output_csv}[/cyan] '
        f'([white]{len(df):,}[/white] rows x [white]{len(df.columns):,}[/white] columns)'
    )


# ------------------------------------------------------------------
# config モード（マルチセグメント対応、video_sync.py から呼ばれる）
# ------------------------------------------------------------------

def pose_recog_from_config(config: dict, model_name: str = DEFAULT_MODEL):
    """設定辞書からポーズ認識を実行する（複数動画セグメント対応）。"""

    import json as _json

    console = Console(legacy_windows=False)
    device  = _get_device()

    output_csv    = config.get('output_csv',    'output.csv')
    synced_start  = int(config.get('synced_start', 0))
    conf_th       = float(config.get('conf_th',    conf_threshold))
    kp_th         = float(config.get('kp_th',      kp_threshold))
    segments      = config.get('video_segments', [])
    save_to       = config.get('save_to', 'pose3d')  # 'pose3d' or 'calib'

    console.print(
        Panel(
            Text('YOLO Pose (config mode)', style='bold cyan', justify='center'),
            border_style='cyan',
            padding=(0, 2),
        )
    )
    console.print(f'[dim]Output CSV   : {output_csv}[/dim]')
    console.print(f'[dim]Synced start : {synced_start}[/dim]')
    console.print(f'[dim]Segments     : {len(segments)}[/dim]')
    console.print(f'[dim]Device       : {device}[/dim]')
    console.print()

    model = YOLO(model_name)

    records         = []
    output_frame_idx = synced_start

    for seg_i, seg in enumerate(segments):

        video_path = seg['path']
        seg_start  = int(seg.get('start', 0))
        seg_end    = seg.get('end', None)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            console.print(f'[bold red]Cannot open:[/bold red] {video_path}')
            continue

        fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if seg_end is None:
            seg_end = frame_num
        seg_end = min(seg_end, frame_num)

        console.print(
            f'[cyan]Segment {seg_i + 1}:[/cyan] {Path(video_path).name}'
            f'  frames {seg_start}-{seg_end}'
        )

        seg_frame_index = seg.get('frame_index', None)
        cur_output_idx  = seg_frame_index if seg_frame_index is not None else output_frame_idx

        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        local_idx = seg_start

        with Progress(
            SpinnerColumn(),
            TextColumn('[bold blue]{task.description}'),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn('*'),
            TimeElapsedColumn(),
            TextColumn('*'),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:

            task = progress.add_task(f'Seg {seg_i + 1}', total=max(0, seg_end - seg_start))

            while local_idx < seg_end:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, device=device, conf=conf_th,
                                        verbose=False, stream=False)
                kps = _extract_keypoints(results[0], width, height)
                records.append(_build_row(cur_output_idx, kps))

                cur_output_idx   += 1
                output_frame_idx += 1
                local_idx        += 1
                progress.advance(task)

        cap.release()

    df = pd.DataFrame(records)

    # キャリブレーション用: 人が未検出のフレーム（全 _v が NaN）を除去
    if save_to == 'calib':
        _v_cols = [c for c in df.columns if c.endswith('_v')]
        if _v_cols:
            _detected_mask = df[_v_cols].notna().any(axis=1)
            _n_all = len(df)
            df = df[_detected_mask].reset_index(drop=True)
            console.print(
                f'[dim]Person detected: {len(df)} / {_n_all} '
                f'(removed {_n_all - len(df)} undetected frames)[/dim]'
            )

    # sync_config.json に直接書き込む
    cam_name = config.get('cam_name')
    if cam_name:
        try:
            from pathlib import Path as _Path2
            _cfg_p = _Path2('sync_config.json')
            _cfg   = _json.loads(_cfg_p.read_text(encoding='utf-8')) if _cfg_p.exists() else {}
            if save_to not in _cfg:
                _cfg[save_to] = {}
            if 'landmarks' not in _cfg[save_to]:
                _cfg[save_to]['landmarks'] = {}
            _cfg[save_to]['landmarks'][cam_name] = {
                'columns': df.columns.tolist(),
                'data':    df.values.tolist(),
            }
            _cfg_p.write_text(
                _json.dumps(_cfg, indent=2, ensure_ascii=False), encoding='utf-8'
            )
            console.print()
            console.print(
                f'[bold green]+[/bold green] Saved [cyan]{cam_name}[/cyan] landmarks '
                f'([white]{len(df):,}[/white] frames) -> [cyan]sync_config.json[/cyan]'
                f'[dim][{save_to}][/dim]'
            )
        except Exception as e:
            console.print(f'[red]ERROR saving to sync_config.json: {e}[/red]')
    else:
        console.print('[yellow]cam_name not set - landmarks not saved[/yellow]')


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_video_names(argv):
    names = []
    for arg in argv:
        if arg.startswith('--'):
            names.append(arg[2:])
        else:
            raise argparse.ArgumentTypeError(f'unrecognized argument: {arg}')
    return names


def resolve_video_paths(folder_path, video_names=None):
    folder = Path(folder_path)
    if video_names:
        video_paths = []
        for name in video_names:
            path = folder / f'{name}.mp4'
            if not path.is_file():
                raise FileNotFoundError(path)
            video_paths.append(path)
        return video_paths
    return sorted(folder.glob('*.mp4'))


if __name__ == '__main__':

    import json as _json

    parser = argparse.ArgumentParser(
        description='Extract pose landmarks from videos using YOLO Pose.',
        epilog=(
            'Examples:\n'
            '  python recog_yolopose.py              # all .mp4 files\n'
            '  python recog_yolopose.py --cam1       # cam1.mp4 only\n'
            '  python recog_yolopose.py --config cam1_config.json\n'
            '  python recog_yolopose.py --model yolov8x-pose.pt'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', metavar='PATH', default=None,
                        help='JSON config file for segment-based processing')
    parser.add_argument('--model',  metavar='NAME', default=DEFAULT_MODEL,
                        help=f'YOLO pose model name or path (default: {DEFAULT_MODEL})')
    args, remaining = parser.parse_known_args()

    console = Console(legacy_windows=False)
    device  = _get_device()

    console.print(f'[dim]Device: {device}[/dim]')
    console.print()

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            console.print(f'[bold red]Config not found:[/bold red] {cfg_path}')
            raise SystemExit(1)
        config = _json.loads(cfg_path.read_text(encoding='utf-8'))
        pose_recog_from_config(config, model_name=args.model)

    else:
        try:
            video_names  = parse_video_names(remaining)
            video_paths  = resolve_video_paths(folder_path, video_names or None)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))
        except FileNotFoundError as e:
            console.print(f'[bold red]Video not found:[/bold red] {e}')
            raise SystemExit(1)

        if not video_paths:
            console.print(f'[bold red]No .mp4 files found in[/bold red] {folder_path}')
            raise SystemExit(1)

        title = 'Selected' if video_names else 'Batch Processing'
        console.print(
            Panel(
                Text(f'{len(video_paths)} video(s)', style='bold white', justify='center'),
                title=f'[bold cyan]{title}[/bold cyan]',
                border_style='cyan',
                padding=(0, 2),
            )
        )
        console.print()

        for i, video_path in enumerate(video_paths, start=1):
            console.rule(f'[bold cyan]{i}/{len(video_paths)}[/bold cyan]  {video_path.name}')
            console.print()
            pose_recog(folder_path, video_path.stem, model_name=args.model)
            if i < len(video_paths):
                console.print()
