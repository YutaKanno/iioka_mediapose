"""
landmarks_3d_nocam5.csv に以下を施して landmarks_3d_processed.csv として保存

  1. 外れ値除去   : セグメント長のローリング MAD で外れ値フレームを NaN 化
  2. スプライン補完: CubicSpline で欠損フレームを埋める
  3. 平滑化       : Savitzky-Golay フィルタ

Usage:
    python process_landmarks_3d.py [--input FILE] [--output FILE] [--drop-first-frames N]
"""

import sys
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io as _io
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from scipy.signal import savgol_filter
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeRemainingColumn,
)

# ── デフォルト値 ────────────────────────────────
_DEFAULT_INPUT_CSV   = 'landmarks_3d_nocam5.csv'
_DEFAULT_OUTPUT_CSV  = 'landmarks_3d_processed.csv'
_DEFAULT_DROP_FRAMES = 500

# ── パラメータ ─────────────────────────────────
OUTLIER_WINDOW    = 15   # ローリング窓（フレーム数）
OUTLIER_THRESH    = 3.5  # MAD の何倍を外れ値とみなすか
SAVGOL_WINDOW     = 11   # Savitzky-Golay 窓（奇数）
SAVGOL_POLY       = 3    # 多項式次数
MAX_INTERP_GAP    = 30   # これより長い連続欠損は補完せず NaN のまま維持

# セグメント長チェック対象（MID_SHOULDERは仮想なので除外）
SEGMENTS = [
    ('NOSE',           'LEFT_EAR'),
    ('NOSE',           'RIGHT_EAR'),
    ('LEFT_SHOULDER',  'RIGHT_SHOULDER'),
    ('LEFT_SHOULDER',  'LEFT_ELBOW'),
    ('LEFT_ELBOW',     'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    ('RIGHT_ELBOW',    'RIGHT_WRIST'),
    ('LEFT_SHOULDER',  'LEFT_HIP'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP'),
    ('LEFT_HIP',       'RIGHT_HIP'),
    ('LEFT_HIP',       'LEFT_KNEE'),
    ('LEFT_KNEE',      'LEFT_ANKLE'),
    ('RIGHT_HIP',      'RIGHT_KNEE'),
    ('RIGHT_KNEE',     'RIGHT_ANKLE'),
]

# ── CLI 引数パース ──────────────────────────────
_parser = argparse.ArgumentParser(
    description = '3D ランドマーク後処理: 外れ値除去・補完・平滑化',
)
_parser.add_argument( '--input',  default = _DEFAULT_INPUT_CSV,
                      help = f'入力 CSV (default: {_DEFAULT_INPUT_CSV})' )
_parser.add_argument( '--output', default = _DEFAULT_OUTPUT_CSV,
                      help = f'出力 CSV (default: {_DEFAULT_OUTPUT_CSV})' )
_parser.add_argument( '--drop-first-frames', type = int, default = _DEFAULT_DROP_FRAMES,
                      metavar = 'N',
                      help = f'先頭 N フレームを除外 (default: {_DEFAULT_DROP_FRAMES})' )
_args, _unknown = _parser.parse_known_args()

INPUT_CSV        = _args.input
OUTPUT_CSV       = _args.output
DROP_FIRST_FRAMES = _args.drop_first_frames

console = Console()

# ── 読み込み ───────────────────────────────────
df = pd.read_csv(INPUT_CSV, encoding='utf-8')
console.print(f'[cyan]Loaded {INPUT_CSV}:[/cyan] {len(df):,} rows')

# 先頭 N フレームを除外
if DROP_FIRST_FRAMES > 0:
    unique_frames = sorted(df['frame'].unique())
    if len(unique_frames) > DROP_FIRST_FRAMES:
        cutoff_frame = unique_frames[DROP_FIRST_FRAMES]
        df = df[df['frame'] >= cutoff_frame].copy()
        console.print(f'[cyan]Dropped first {DROP_FIRST_FRAMES} frames → remaining:[/cyan] {df["frame"].nunique():,} frames')
    else:
        console.print(f'[yellow]drop-first-frames={DROP_FIRST_FRAMES} but only {len(unique_frames)} unique frames → skipping drop[/yellow]')
else:
    console.print(f'[dim]drop-first-frames=0 → no frames dropped[/dim]')

frames   = sorted(df['frame'].unique())
n_frames = len(frames)

# ── ワイド形式に変換 ────────────────────────────
px = df.pivot(index='frame', columns='landmark', values='x').reindex(frames)
py = df.pivot(index='frame', columns='landmark', values='y').reindex(frames)
pz = df.pivot(index='frame', columns='landmark', values='z').reindex(frames)

# ── 1. セグメント長による外れ値除去 ─────────────
console.print('[cyan]Detecting outliers by segment length...[/cyan]')
outlier_counts = 0

for a, b in SEGMENTS:
    if a not in px.columns or b not in px.columns:
        continue

    length = np.sqrt(
        (px[a] - px[b])**2 +
        (py[a] - py[b])**2 +
        (pz[a] - pz[b])**2
    )

    roll_med = length.rolling(OUTLIER_WINDOW, center=True, min_periods=3).median()
    roll_mad = (
        (length - roll_med).abs()
        .rolling(OUTLIER_WINDOW, center=True, min_periods=3)
        .median()
    )
    global_mad = float((length - length.median()).abs().median())
    roll_mad = roll_mad.replace(0, np.nan).ffill().bfill().fillna(global_mad + 1e-8)
    z_score = (length - roll_med).abs() / (roll_mad * 1.4826 + 1e-8)

    outlier_mask = z_score > OUTLIER_THRESH
    n_out = int(outlier_mask.sum())
    if n_out > 0:
        console.print(f'  [yellow]{a}–{b}:[/yellow] {n_out} outlier frames')
        outlier_counts += n_out

    for coord_df in [px, py, pz]:
        coord_df.loc[outlier_mask, a] = np.nan
        coord_df.loc[outlier_mask, b] = np.nan

console.print(f'[cyan]Total outlier frames marked:[/cyan] {outlier_counts}')

# ── 2 & 3. スプライン補完 + Savitzky-Golay 平滑化 ──
def process_series(s: np.ndarray) -> np.ndarray:
    s = s.copy().astype(float)
    idx   = np.arange(n_frames)
    valid = ~np.isnan(s)
    n_valid = int(valid.sum())

    if n_valid >= 4:
        interp = PchipInterpolator(idx[valid], s[valid], extrapolate=False)
        # MAX_INTERP_GAP 以下の欠損ギャップのみ補完
        nan_idx = np.where(~valid)[0]
        if len(nan_idx) > 0:
            # 連続ギャップをグループ化
            gaps = np.split(nan_idx, np.where(np.diff(nan_idx) > 1)[0] + 1)
            for gap in gaps:
                if len(gap) <= MAX_INTERP_GAP:
                    filled = interp(gap)
                    # extrapolate=False なので範囲外は NaN → そのまま維持
                    mask = ~np.isnan(filled)
                    s[gap[mask]] = filled[mask]
    elif n_valid >= 2:
        # 短い系列は線形補間（MAX_INTERP_GAP 制限付き）
        ser = pd.Series(s)
        s = ser.interpolate('linear', limit=MAX_INTERP_GAP).values

    # NaN が残っていても Savitzky-Golay は NaN のないセグメントのみ適用
    result = s.copy()
    not_nan = ~np.isnan(s)
    segments = np.split(np.where(not_nan)[0], np.where(np.diff(np.where(not_nan)[0]) > 1)[0] + 1)
    win = SAVGOL_WINDOW
    for seg in segments:
        if len(seg) >= win and win >= SAVGOL_POLY + 2:
            result[seg] = savgol_filter(s[seg], win, SAVGOL_POLY)

    return result

landmarks = sorted(px.columns)
out_parts = []

with Progress(
    SpinnerColumn(),
    TextColumn('[bold blue]{task.description}'),
    BarColumn(bar_width=40),
    TextColumn('{task.completed}/{task.total}'),
    TimeRemainingColumn(),
    console=console,
) as progress:
    task = progress.add_task('Interpolating & smoothing', total=len(landmarks))

    for lm in landmarks:
        result = pd.DataFrame({
            'frame':    frames,
            'landmark': lm,
            'x': process_series(px[lm].values),
            'y': process_series(py[lm].values),
            'z': process_series(pz[lm].values),
        })
        out_parts.append(result)
        progress.advance(task)

# ── 保存 ───────────────────────────────────────
df_out = pd.concat(out_parts, ignore_index=True)[['frame', 'landmark', 'x', 'y', 'z']]
df_out = df_out.sort_values(['frame', 'landmark']).reset_index(drop=True)
df_out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
console.print(
    f'[bold green]✓[/bold green] Saved [cyan]{OUTPUT_CSV}[/cyan]'
    f'  ({len(df_out):,} rows, {df_out["frame"].nunique():,} frames)'
)
