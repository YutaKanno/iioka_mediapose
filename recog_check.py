# %%
import os
import cv2
import numpy as np
import pandas as pd

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

console = Console()

csv_path = 'cam1.csv'
video_path = csv_path.replace( '.csv', '.mp4' )

df = pd.read_csv( csv_path )

output_folder = 'frames'
output_mp4    = csv_path.replace( '.csv', '_skeleton.mp4' )
frame_ext     = 'jpg'
img_width     = 640
img_height    = 480

os.makedirs( output_folder, exist_ok = True )

console.print(
    Panel(
        Text( 'Pose Skeleton Visualization', style = 'bold cyan', justify = 'center' ),
        border_style = 'cyan',
        padding = ( 0, 2 ),
    )
)


# --------------------------------------------------
# Landmark Names
# --------------------------------------------------

landmark_names = [ 
    name.split( '_x' )[ 0 ] 
    for name in df.columns 
    if name.endswith( '_x' )
]

x_columns = [ f'{name}_x' for name in landmark_names ]
y_columns = [ f'{name}_y' for name in landmark_names ]


# --------------------------------------------------
# Inverse Y Axis
# --------------------------------------------------

ymax = df[ y_columns ].max().max()
df[ y_columns ] = ymax - df[ y_columns ]


# --------------------------------------------------
# Calc Add Points (after Y inversion)
# --------------------------------------------------

df[ 'MID_SHOULDER_x' ] = ( df[ 'RIGHT_SHOULDER_x' ] + df[ 'LEFT_SHOULDER_x' ] ) / 2
df[ 'MID_SHOULDER_y' ] = ( df[ 'RIGHT_SHOULDER_y' ] + df[ 'LEFT_SHOULDER_y' ] ) / 2
df[ 'MID_HIP_x' ] = ( df[ 'RIGHT_HIP_x' ] + df[ 'LEFT_HIP_x' ] ) / 2
df[ 'MID_HIP_y' ] = ( df[ 'RIGHT_HIP_y' ] + df[ 'LEFT_HIP_y' ] ) / 2

plot_x_columns = x_columns + [ 'MID_SHOULDER_x', 'MID_HIP_x' ]
plot_y_columns = y_columns + [ 'MID_SHOULDER_y', 'MID_HIP_y' ]


# --------------------------------------------------
# Line Definition
# --------------------------------------------------

line_def = [

    ( 'RIGHT_INDEX', 'RIGHT_WRIST' ),
    ( 'RIGHT_WRIST', 'RIGHT_ELBOW' ),
    ( 'RIGHT_ELBOW', 'RIGHT_SHOULDER' ),

    ( 'LEFT_INDEX', 'LEFT_WRIST' ),
    ( 'LEFT_WRIST', 'LEFT_ELBOW' ),
    ( 'LEFT_ELBOW', 'LEFT_SHOULDER' ),

    ( 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL' ),
    ( 'RIGHT_HEEL',       'RIGHT_ANKLE' ),
    ( 'RIGHT_ANKLE',      'RIGHT_KNEE' ),
    ( 'RIGHT_KNEE',       'RIGHT_HIP' ),

    ( 'LEFT_FOOT_INDEX', 'LEFT_HEEL' ),
    ( 'LEFT_HEEL',       'LEFT_ANKLE' ),
    ( 'LEFT_ANKLE',      'LEFT_KNEE' ),
    ( 'LEFT_KNEE',       'LEFT_HIP' ),

    ( 'RIGHT_EYE', 'RIGHT_EAR' ),
    ( 'LEFT_EYE', 'LEFT_EAR' ),

    ( 'RIGHT_EYE',      'LEFT_EYE' ),
    ( 'RIGHT_EAR',      'LEFT_EAR' ),
    ( 'RIGHT_SHOULDER', 'LEFT_SHOULDER' ),
    ( 'RIGHT_HIP',      'LEFT_HIP' ),

    ( 'NOSE',         'MID_SHOULDER' ),
    ( 'MID_SHOULDER', 'MID_HIP' ),

]


def make_segments( df_frame, line_def ):

    segments = []

    for line in line_def:

        x0 = df_frame[ f'{line[0]}_x' ]
        y0 = df_frame[ f'{line[0]}_y' ]
        x1 = df_frame[ f'{line[1]}_x' ]
        y1 = df_frame[ f'{line[1]}_y' ]

        if pd.isna( [ x0, y0, x1, y1 ] ).any():
            continue

        segments.append( ( float( x0 ), float( y0 ), float( x1 ), float( y1 ) ) )

    return segments


def draw_frame( segments, xmin, xmax, ymin, ymax, width, height ):

    canvas = np.full( ( height, width, 3 ), 255, dtype = np.uint8 )

    x_scale = ( width  - 1 ) / ( xmax - xmin )
    y_scale = ( height - 1 ) / ( ymax - ymin )

    for x0, y0, x1, y1 in segments:

        px0 = int( ( x0 - xmin ) * x_scale )
        py0 = int( ( ymax - y0 ) * y_scale )
        px1 = int( ( x1 - xmin ) * x_scale )
        py1 = int( ( ymax - y1 ) * y_scale )

        cv2.line( canvas, ( px0, py0 ), ( px1, py1 ), ( 0, 0, 0 ), 1, cv2.LINE_AA )

    return canvas


# --------------------------------------------------
# Frame Plot + MP4
# --------------------------------------------------

xmin, xmax = df[ plot_x_columns ].min().min(), df[ plot_x_columns ].max().max()
ymin, ymax = df[ plot_y_columns ].min().min(), df[ plot_y_columns ].max().max()

cap = cv2.VideoCapture( video_path )
fps = cap.get( cv2.CAP_PROP_FPS )
cap.release()

if fps > 0:
    fps_label = f'{fps:.2f} (from video)'
else:
    fps = 30.0
    fps_label = f'{fps:.2f} (default)'

info_table = Table( title = 'Input / Output', show_header = False, border_style = 'dim' )
info_table.add_column( 'Key', style = 'bold green' )
info_table.add_column( 'Value', style = 'white' )
info_table.add_row( 'CSV', csv_path )
info_table.add_row( 'Frames', f'{len( df ):,}' )
info_table.add_row( 'Canvas', f'{img_width} x {img_height}' )
info_table.add_row( 'FPS', fps_label )
info_table.add_row( 'Frame dir', output_folder )
info_table.add_row( 'Output MP4', output_mp4 )
console.print( info_table )
console.print()

writer = cv2.VideoWriter(
    output_mp4,
    cv2.VideoWriter_fourcc( *'mp4v' ),
    fps,
    ( img_width, img_height ),
)

drawn_count = 0

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

    task = progress.add_task( 'Rendering frames', total = len( df ) )

    for frame_idx in range( len( df ) ):

        segments = make_segments( df.iloc[ frame_idx ], line_def )

        if segments:
            drawn_count += 1

        canvas = draw_frame( segments, xmin, xmax, ymin, ymax, img_width, img_height )

        cv2.imwrite(
            os.path.join( output_folder, f'frame_{frame_idx:04d}.{frame_ext}' ),
            canvas,
            [ cv2.IMWRITE_JPEG_QUALITY, 85 ],
        )
        writer.write( canvas )
        progress.advance( task )

writer.release()

summary_table = Table( title = 'Summary', show_header = False, border_style = 'dim' )
summary_table.add_column( 'Key', style = 'bold magenta' )
summary_table.add_column( 'Value', style = 'white' )
summary_table.add_row( 'Rendered frames', f'{len( df ):,}' )
summary_table.add_row( 'Frames with skeleton', f'{drawn_count:,}' )
summary_table.add_row( 'Skeleton rate', f'{drawn_count / len( df ) * 100:.1f}%' )
summary_table.add_row( 'Frame images', output_folder )
summary_table.add_row( 'Output MP4', output_mp4 )
console.print()
console.print( summary_table )
console.print()
console.print(
    f'[bold green]✓[/bold green] Saved [cyan]{output_mp4}[/cyan] '
    f'([white]{len( df ):,}[/white] frames, [white]{fps:.2f}[/white] fps)'
)
# %%
