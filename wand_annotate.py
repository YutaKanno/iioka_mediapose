"""
Step 1a: Wand calibration annotation tool

Each camera video (cam1.mp4 ~ cam6.mp4) contains wand footage for 3 poses in sequence.
For each pose, select one still frame and click 4 points on the wand
(0.0m, 0.5m, 1.0m, 1.5m) with the mouse. Results are saved to CSV.

Controls:
    [a] / [d] : Move to previous / next frame
    [c]       : Start click mode (click in order: 0.0m -> 0.5m -> 1.0m -> 1.5m)
    [r]       : Reset clicks
    After 4 clicks, automatically advances to the next pose.

Before running, update video_dir in the settings section below.
"""

import os
import cv2
import pandas as pd

# ------------------------------
# Settings
# ------------------------------
video_dir = '/Users/yutakanno/Library/CloudStorage/GoogleDrive-poohyuta604@gmail.com/My Drive/mediapipe_test'  # folder containing cam1.mp4 ~ cam6.mp4
camera_names = ["cam3", "cam4", "cam5", "cam6", "cam1", "cam2"]
pose_names = ["pose1", "pose2", "pose3"]
point_labels = ["0.0m", "0.5m", "1.0m", "1.5m"]
output_csv_path = "./wand_annotations.csv"

clicked_points = []
click_mode = False
display_scale = 1


def on_mouse_click(event, x, y, flags, param):
    if not click_mode:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            ox, oy = x // display_scale, y // display_scale
            clicked_points.append((ox, oy))
            label = point_labels[len(clicked_points) - 1]
            print(f"  Clicked {label}: (u={ox}, v={oy})")


def get_frame(video_capture, frame_number):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame_image = video_capture.read()
    if not success:
        return None
    return frame_image


def draw_clicked_points(frame_image):
    image_with_points = frame_image.copy()
    for point_index in range(len(clicked_points)):
        x, y = clicked_points[point_index]
        cv2.circle(image_with_points, (x, y), 4, (0, 0, 255), -1)
        label = point_labels[point_index]
        cv2.putText(image_with_points, label, (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return image_with_points


def annotate_one_pose(video_path, camera_name, pose_name):
    global clicked_points
    global click_mode

    global display_scale
    DISPLAY_SCALE = 3
    display_scale = DISPLAY_SCALE

    video_capture = cv2.VideoCapture(video_path)
    total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    window_title = f"{camera_name} / {pose_name}  [a/d]:prev/next  [g]:go to frame  [c]:click mode  [r]:reset"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, on_mouse_click)

    current_frame_number = 0
    click_mode = False
    clicked_points = []

    while True:
        frame_image = get_frame(video_capture, current_frame_number)
        if frame_image is None:
            break

        display_image = draw_clicked_points(frame_image)
        h, w = display_image.shape[:2]
        display_image = cv2.resize(display_image, (w * DISPLAY_SCALE, h * DISPLAY_SCALE),
                                   interpolation=cv2.INTER_LINEAR)
        status_text = f"frame: {current_frame_number}/{total_frame_count - 1}"
        cv2.putText(display_image, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        if click_mode:
            next_idx = len(clicked_points)
            if next_idx < 4:
                next_label = point_labels[next_idx]
                mode_text = f"[CLICK MODE] Next: {next_label} ({next_idx + 1}/4)"
            else:
                mode_text = "All 4 points done"
            cv2.putText(display_image, mode_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
        else:
            cv2.putText(display_image, "Press [c] to start click mode", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)
        cv2.imshow(window_title, display_image)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("a"):
            current_frame_number = max(current_frame_number - 1, 0)
        elif key == ord("d"):
            current_frame_number = min(current_frame_number + 1, total_frame_count - 1)
        elif key == ord("g"):
            user_input = input(f"Go to frame (0 - {total_frame_count - 1}): ")
            if user_input.isdigit():
                current_frame_number = max(0, min(int(user_input), total_frame_count - 1))
        elif key == ord("c"):
            click_mode = True
            print(f"[{camera_name} / {pose_name}] Click mode started. "
                  f"Click in order: 0.0m -> 0.5m -> 1.0m -> 1.5m")
        elif key == ord("r"):
            clicked_points = []
            print("Clicks reset.")

        if click_mode and len(clicked_points) == 4:
            break

    video_capture.release()
    cv2.destroyWindow(window_title)
    cv2.waitKey(1)  # flush event loop on macOS to prevent freeze after window close

    annotation_rows = []
    for point_index in range(len(clicked_points)):
        x, y = clicked_points[point_index]
        annotation_rows.append({
            "camera": camera_name,
            "pose": pose_name,
            "point_label": point_labels[point_index],
            "frame": current_frame_number,
            "u": x,
            "v": y,
        })
    return annotation_rows


# ------------------------------
# Main: loop over all cameras x all poses
# ------------------------------
def main():
    all_annotation_rows = []

    for camera_name in camera_names:
        video_path = os.path.join(video_dir, f"{camera_name}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found. Skipping.")
            continue

        for pose_name in pose_names:
            rows = annotate_one_pose(video_path, camera_name, pose_name)
            all_annotation_rows.extend(rows)

    annotation_dataframe = pd.DataFrame(all_annotation_rows)
    annotation_dataframe.to_csv(output_csv_path, index=False)
    print(f"Annotations saved to {output_csv_path}")


if __name__ == "__main__":
    main()
