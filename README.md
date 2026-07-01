# iioka_mediapipe

MediaPipe Pose Landmarker を使った動画からの骨格ランドマーク抽出・可視化ツール。

## セットアップ

```bash
pip install -r requirements.txt
```

`pose_landmarker_heavy.task` を同ディレクトリに配置してください（同梱済み）。

## 使い方

### 1. ランドマーク抽出 (`recog.py`)

`recog.py` 内の `folder_path` と `video_name` を編集し、動画からランドマークを CSV に出力します。

```bash
python recog.py
```

出力: `{video_name}.csv`

### 2. スケルトン可視化 (`recog_check.py`)

CSV からフレーム画像とスケルトン動画を生成します。

```bash
python recog_check.py
```

出力: `frames/` フォルダ、`{video_name}_skeleton.mp4`

## 設定

`recog.py` の先頭で検出閾値を調整できます。

- `min_pose_detection_confidence`
- `min_pose_presence_confidence`
- `min_tracking_confidence`
- `visibility_threshold`
