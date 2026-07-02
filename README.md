# iioka_mediapose

多カメラ映像の同期・カメラキャリブレーション・MediaPipe による姿勢推定・3次元座標算出をまとめて行う GUI アプリケーション。

## セットアップ

```bash
pip install -r requirements.txt
```

[MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) のモデルファイル（`pose_landmarker_heavy.task`）を同ディレクトリに配置してください。

## 使い方

```bash
python video_sync.py
```

## ワークフロー

| Step | タブ名 | 内容 |
|------|--------|------|
| 1 | Sync | 複数カメラ映像の同期（フレームオフセット調整） |
| 2 | Calibration | ワンドアノテーション CSV を用いたカメラキャリブレーション |
| 3 | Results | キャリブレーション結果の確認（再投影誤差など） |
| 4 | Pose Recognition | MediaPipe によるポーズランドマーク抽出（全カメラ一括） |
| 5 | Stick Check | スティックフィギュアの確認・カメラ別 visibility 閾値設定 |
| 6 | 3D Recon | 三角測量による 3D 座標算出 → Plotly インタラクティブ表示 |

## ファイル構成

| ファイル | 役割 |
|----------|------|
| `video_sync.py` | メイン GUI アプリ |
| `est_camera_poses.py` | カメラキャリブレーション（ワンド法） |
| `recog_mediapipe.py` | MediaPipe Pose Landmarker による姿勢推定 |
| `triangulate_only.py` | 多視点三角測量（DLT） |
| `process_landmarks_3d.py` | 3D ランドマークの後処理 |
| `make_3d_plot.py` | Plotly による 3D スティックフィギュア生成 |
| `wand_annotate.py` | キャリブレーションワンドのアノテーションツール |
| `wand_detection.py` | ワンドマーカーの自動検出 |
| `recog_check.py` | 認識結果のスケルトン動画生成 |
