"""
3D landmark stick-figure animation
3 views: XY (front), XZ (top), YZ (side)
Output: stick_figure_3d.mp4

Coordinate system (cam4 origin):
  X  = cam4's right (1st-base side)
  Y  = cam4's down  → displayed as -Y (up = positive)
  Z  = cam4's forward (toward pitcher / outfield)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn, TextColumn

INPUT_CSV  = 'landmarks_3d_processed.csv'
OUTPUT_MP4 = 'stick_figure_3d.mp4'
FPS        = 30

# スティックの骨格接続（ランドマーク名）
# MID_SHOULDER は LEFT/RIGHT_SHOULDER の中点（仮想ランドマーク）
CONNECTIONS = [
    ('NOSE',            'LEFT_EAR'),
    ('NOSE',            'RIGHT_EAR'),
    ('NOSE',            'MID_SHOULDER'),
    ('LEFT_SHOULDER',   'RIGHT_SHOULDER'),
    ('LEFT_SHOULDER',   'LEFT_ELBOW'),
    ('LEFT_ELBOW',      'LEFT_WRIST'),
    ('RIGHT_SHOULDER',  'RIGHT_ELBOW'),
    ('RIGHT_ELBOW',     'RIGHT_WRIST'),
    ('LEFT_SHOULDER',   'LEFT_HIP'),
    ('RIGHT_SHOULDER',  'RIGHT_HIP'),
    ('LEFT_HIP',        'RIGHT_HIP'),
    ('LEFT_HIP',        'LEFT_KNEE'),
    ('LEFT_KNEE',       'LEFT_ANKLE'),
    ('RIGHT_HIP',       'RIGHT_KNEE'),
    ('RIGHT_KNEE',      'RIGHT_ANKLE'),
]

# ────────────────────────────────────────────────
# Load & pivot
# ────────────────────────────────────────────────

console = Console()
console.print( '[bold cyan]Loading landmarks...[/bold cyan]' )

df     = pd.read_csv( INPUT_CSV )
frames = sorted( df['frame'].unique() )

px = df.pivot( index='frame', columns='landmark', values='x' ).reindex( frames )
py = df.pivot( index='frame', columns='landmark', values='y' ).reindex( frames )
pz = df.pivot( index='frame', columns='landmark', values='z' ).reindex( frames )

available = set( px.columns )

# MID_SHOULDER を仮想ランドマークとして追加
if 'LEFT_SHOULDER' in available and 'RIGHT_SHOULDER' in available:
    px['MID_SHOULDER'] = ( px['LEFT_SHOULDER'] + px['RIGHT_SHOULDER'] ) / 2
    py['MID_SHOULDER'] = ( py['LEFT_SHOULDER'] + py['RIGHT_SHOULDER'] ) / 2
    pz['MID_SHOULDER'] = ( pz['LEFT_SHOULDER'] + pz['RIGHT_SHOULDER'] ) / 2
    available.add('MID_SHOULDER')

connections = [ ( a, b ) for a, b in CONNECTIONS if a in available and b in available ]

# ────────────────────────────────────────────────
# Axis limits  (Y is flipped for display: -Y = up)
# ────────────────────────────────────────────────

def _lim( arr, pad = 0.3 ):
    v = arr[ ~np.isnan( arr ) ]
    return v.min() - pad, v.max() + pad

Xall = px.values.ravel()
Yall = ( -py ).values.ravel()  # flip
Zall = pz.values.ravel()

xlim = _lim( Xall )
ylim = _lim( Yall )
zlim = _lim( Zall )

planes = [
    ( 'XY  (front)',  'X →',  '↑ -Y (up)',  'x',  xlim, ylim ),
    ( 'XZ  (top)',    'X →',  'Z (depth)',   'xz', xlim, zlim ),
    ( 'YZ  (side)',   '↑ -Y', 'Z (depth)',   'yz', ylim, zlim ),
]

# ────────────────────────────────────────────────
# Figure setup
# ────────────────────────────────────────────────

fig, axes = plt.subplots( 1, 3, figsize = ( 15, 5 ) )
fig.patch.set_facecolor( '#111111' )
fig.suptitle( 'Pitcher 3D Pose', color = 'white', fontsize = 13 )

line_objs = []
dot_objs  = []

for ax, ( title, xlabel, ylabel, _, xlim_, ylim_ ) in zip( axes, planes ):
    ax.set_facecolor( '#1a1a1a' )
    ax.set_title( title, color = 'white', fontsize = 10 )
    ax.set_xlabel( xlabel, color = '#aaaaaa', fontsize = 8 )
    ax.set_ylabel( ylabel, color = '#aaaaaa', fontsize = 8 )
    ax.set_xlim( xlim_ )
    ax.set_ylim( ylim_ )
    ax.tick_params( colors = '#666666', labelsize = 7 )
    for spine in ax.spines.values():
        spine.set_color( '#444444' )
    ax.set_aspect( 'equal', adjustable = 'datalim' )
    ax.grid( True, color = '#333333', linewidth = 0.5 )

    lines = [ ax.plot( [], [], color = '#00ccff', lw = 1.5, solid_capstyle = 'round' )[ 0 ]
              for _ in connections ]
    dots  = ax.plot( [], [], 'o', color = 'white', ms = 3, zorder = 5 )[ 0 ]
    line_objs.append( lines )
    dot_objs.append( dots )

frame_label = fig.text( 0.5, 0.01, '', ha = 'center', color = '#888888', fontsize = 9 )

plt.tight_layout( rect = [ 0, 0.03, 1, 0.95 ] )

# ────────────────────────────────────────────────
# Animation update
# ────────────────────────────────────────────────

def _get( series, name ):
    try:
        v = series[ name ]
        return np.nan if pd.isna( v ) else float( v )
    except KeyError:
        return np.nan


def update( fi ):
    frame = frames[ fi ]
    x  =  px.loc[ frame ]
    y  = -py.loc[ frame ]   # flip Y
    z  =  pz.loc[ frame ]

    frame_label.set_text( f'frame {frame}' )

    coords = [
        ( x, y ),   # XY
        ( x, z ),   # XZ
        ( y, z ),   # YZ
    ]

    artists = []

    for pi, ( h_ser, v_ser ) in enumerate( coords ):
        for li, ( a, b ) in enumerate( connections ):
            ha, hb = _get( h_ser, a ), _get( h_ser, b )
            va, vb = _get( v_ser, a ), _get( v_ser, b )
            if np.isnan( ha ) or np.isnan( hb ) or np.isnan( va ) or np.isnan( vb ):
                line_objs[ pi ][ li ].set_data( [], [] )
            else:
                line_objs[ pi ][ li ].set_data( [ ha, hb ], [ va, vb ] )
            artists.append( line_objs[ pi ][ li ] )

        all_lm = set( a for a, _ in connections ) | set( b for _, b in connections )
        hs = np.array( [ _get( h_ser, lm ) for lm in all_lm ] )
        vs = np.array( [ _get( v_ser, lm ) for lm in all_lm ] )
        mask = ~( np.isnan( hs ) | np.isnan( vs ) )
        dot_objs[ pi ].set_data( hs[ mask ], vs[ mask ] )
        artists.append( dot_objs[ pi ] )

    artists.append( frame_label )
    return artists


# ────────────────────────────────────────────────
# Render
# ────────────────────────────────────────────────

console.print( f'[bold cyan]Rendering {len(frames)} frames → {OUTPUT_MP4}[/bold cyan]' )

anim = animation.FuncAnimation(
    fig,
    update,
    frames   = len( frames ),
    interval = 1000 / FPS,
    blit     = True,
)

writer = animation.FFMpegWriter( fps = FPS, bitrate = 2000,
                                 extra_args = [ '-pix_fmt', 'yuv420p' ] )

with Progress(
    SpinnerColumn(),
    TextColumn( '[bold blue]{task.description}' ),
    BarColumn( bar_width = 40 ),
    MofNCompleteColumn(),
    TextColumn( '•' ),
    TimeRemainingColumn(),
    console = console,
) as progress:
    task = progress.add_task( 'Encoding', total = len( frames ) )

    def _cb( i, n ):
        progress.update( task, completed = i + 1 )

    anim.save( OUTPUT_MP4, writer = writer, dpi = 120, progress_callback = _cb )

console.print( f'[bold green]✓[/bold green] Saved [cyan]{OUTPUT_MP4}[/cyan]' )
