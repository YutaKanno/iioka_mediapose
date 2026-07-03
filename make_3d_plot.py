"""
landmarks_3d_processed.csv → Plotly インタラクティブ 3D スティックフィギュア
出力: stick_figure_3d.html

Usage:
    python make_3d_plot.py [--input FILE] [--output FILE] [--frame-step N]
"""

import sys
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rich.console import Console

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io as _io
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_DEFAULT_INPUT_CSV   = 'landmarks_3d_processed.csv'
_DEFAULT_OUTPUT_HTML = 'stick_figure_3d.html'
_DEFAULT_FRAME_STEP  = 1

_parser = argparse.ArgumentParser( description = '3D スティックフィギュア Plotly HTML 生成' )
_parser.add_argument( '--input',      default = _DEFAULT_INPUT_CSV,
                      help = f'入力 CSV (default: {_DEFAULT_INPUT_CSV})' )
_parser.add_argument( '--output',     default = _DEFAULT_OUTPUT_HTML,
                      help = f'出力 HTML (default: {_DEFAULT_OUTPUT_HTML})' )
_parser.add_argument( '--frame-step', type = int, default = _DEFAULT_FRAME_STEP,
                      metavar = 'N',
                      help = f'フレームステップ (default: {_DEFAULT_FRAME_STEP})' )
_args, _unknown = _parser.parse_known_args()

INPUT_CSV       = _args.input
OUTPUT_HTML     = _args.output
PLOT_FRAME_STEP = _args.frame_step

CONNECTIONS = [
    ('NOSE',           'LEFT_EAR'),
    ('NOSE',           'RIGHT_EAR'),
    ('NOSE',           'MID_SHOULDER'),
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

console = Console()

# ── データ読み込み ──────────────────────────────────────
df3d = pd.read_csv(INPUT_CSV, encoding='utf-8')
console.print(f'[cyan]Loaded {INPUT_CSV}:[/cyan] {len(df3d):,} rows')

plot_frames = sorted(df3d['frame'].unique())[::PLOT_FRAME_STEP]
console.print(f'[cyan]Frames to plot:[/cyan] {len(plot_frames)} (step={PLOT_FRAME_STEP})')

# ── 軸範囲（5–95パーセンタイルで外れ値除去）──────────────
all_x =  df3d['x'].values
all_y = -df3d['y'].values   # Y反転（上が正）
all_z =  df3d['z'].values

def axis_range(v, pad=0.3):
    v = v[~np.isnan(v)]
    lo, hi = float(np.percentile(v, 5)), float(np.percentile(v, 95))
    span = hi - lo
    return [lo - span * pad, hi + span * pad]

xrange = [-10, 10]
yrange = axis_range(all_y)
zrange = axis_range(all_z)
console.print(f'X: {xrange[0]:.1f} – {xrange[1]:.1f}  '
              f'Y: {yrange[0]:.1f} – {yrange[1]:.1f}  '
              f'Z: {zrange[0]:.1f} – {zrange[1]:.1f}')

# 物理スケールを統一: 各軸のspan比でaspectratioを計算
_xspan = xrange[1] - xrange[0]
_yspan = yrange[1] - yrange[0]
_zspan = zrange[1] - zrange[0]
_max_span = max(_xspan, _yspan, _zspan)
aspect_ratio = dict(x=_xspan/_max_span, y=_yspan/_max_span, z=_zspan/_max_span)

# ── フレームトレース生成 ────────────────────────────────
def frame_traces(frame_id):
    sub = df3d[df3d['frame'] == frame_id].set_index('landmark')

    def get(lm):
        if lm == 'MID_SHOULDER':
            ls = get('LEFT_SHOULDER')
            rs = get('RIGHT_SHOULDER')
            if ls and rs:
                return ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2)
            return None
        if lm not in sub.index:
            return None
        r = sub.loc[lm]
        return float(r['x']), -float(r['y']), float(r['z'])

    bx, by, bz = [], [], []
    for a, b in CONNECTIONS:
        pa, pb = get(a), get(b)
        if pa and pb:
            bx += [pa[0], pb[0], None]
            by += [pa[1], pb[1], None]
            bz += [pa[2], pb[2], None]

    jx, jy, jz, jlabels = [], [], [], []
    for lm in sub.index:
        r = sub.loc[lm]
        jx.append(float(r['x']))
        jy.append(-float(r['y']))
        jz.append(float(r['z']))
        jlabels.append(lm)

    return [
        go.Scatter3d(x=bx, y=by, z=bz,
                     mode='lines',
                     line=dict(color='black', width=4),
                     hoverinfo='none', name='bones'),
        go.Scatter3d(x=jx, y=jy, z=jz,
                     mode='markers',
                     marker=dict(size=4, color='white'),
                     text=jlabels,
                     hovertemplate='%{text}<br>x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>',
                     name='joints'),
    ]

# ── アニメーション組み立て ──────────────────────────────
console.print('[cyan]Building frames...[/cyan]')
init_traces = frame_traces(plot_frames[0])
anim_frames = [go.Frame(data=frame_traces(fid), name=str(fid)) for fid in plot_frames]

def fixed_axis(title, rng):
    return dict(title=title, range=rng, autorange=False,
                gridcolor='#444', color='white')

layout = go.Layout(
    title=dict(text='Pitcher 3D Pose  (cam5 excluded)', font=dict(color='white')),
    paper_bgcolor='#111111',
    uirevision='constant',
    scene=dict(
        bgcolor='#1a1a1a',
        xaxis=fixed_axis('X (3塁←→1塁)', xrange),
        yaxis=fixed_axis('-Y (↑ up)',      yrange),
        zaxis=fixed_axis('Z (depth)',       zrange[::-1]),
        aspectmode='manual',
        aspectratio=aspect_ratio,
        camera=dict(
            eye=dict(x=0, y=0, z=2.5),
            up=dict(x=0, y=1, z=0),
        ),
    ),
    updatemenus=[dict(
        type='buttons', showactive=False, y=0.02, x=0.1,
        buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=33, redraw=True),
                                  fromcurrent=True, mode='immediate')]),
            dict(label='⏸ Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode='immediate')]),
        ],
    )],
    sliders=[dict(
        active=0,
        currentvalue=dict(prefix='frame: ', font=dict(color='white')),
        pad=dict(t=50),
        steps=[
            dict(method='animate',
                 args=[[str(fid)], dict(mode='immediate',
                                        frame=dict(duration=0, redraw=True))],
                 label=str(fid))
            for fid in plot_frames
        ],
    )],
    font=dict(color='white'),
    legend=dict(font=dict(color='white')),
)

fig = go.Figure(data=init_traces, frames=anim_frames, layout=layout)
fig.write_html(OUTPUT_HTML, include_plotlyjs='cdn')
console.print(
    f'[bold green]✓[/bold green] Saved [cyan]{OUTPUT_HTML}[/cyan]'
    f'  ({len(plot_frames):,} frames)'
)
