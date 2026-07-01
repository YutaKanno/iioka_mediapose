import os
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
video_name  = 'cam1'

model_path = Path( __file__ ).parent / 'pose_landmarker_heavy.task'

# 6台中2台で足りる前提 → 各カメラは厳しめに弾く
min_pose_detection_confidence = 0.8
min_pose_presence_confidence  = 0.8
min_tracking_confidence       = 0.8
visibility_threshold          = 0.95


def pose_recog( folder_path, video_name, model_path ):

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


if __name__ == '__main__':
    pose_recog( folder_path, video_name, model_path )