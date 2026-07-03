"""
6カメラ動画同期ツール
Step 1: Sync        - 各カメラのイベントフレームを揃えて同期
Step 2: Calibration - wand_annotations.csv の読み込みとキャリブレーション実行
Step 3: Results     - カメラ配置・再投影誤差の可視化
Step 4: Pose 3D     - MediaPipe ポーズ抽出 → 三角測量 → 3D 可視化
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# Windows コンソールを UTF-8 に強制（UnicodeEncodeError 防止）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io as _io
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# FFmpeg のフレームスレッディングを無効化（H264 マルチスレッド時の double-free 防止）
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'threads;1')

import cv2
import numpy as np
from PIL import Image, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

# ── 定数 ───────────────────────────────────────────────
N_CAMS       = 6
COLS         = 3
PANEL_W      = 360
PANEL_H      = 180   # main() で画面サイズに合わせて上書き
CLICK_RADIUS = 12

_ANSI_RE = re.compile(r'\x1b(?:\[[0-9;?]*[a-zA-Z]|\([0-9A-Za-z])')


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub('', s)


# ══════════════════════════════════════════════════════
class VideoPanel:
    def __init__(self, parent: ttk.Frame, cam_idx: int):
        self.cam_idx        = cam_idx
        self.caps: list[cv2.VideoCapture] = []
        self.paths: list[str]             = []
        self.segment_frames: list[int]    = []
        self.cum_frames: list[int]        = [0]
        self.total_frames   = 0
        self.current_frame  = 0
        self.event_frame    = tk.IntVar(value=0)
        self._photo         = None
        self._frame_req: int | None = None   # 最新リクエストフレーム番号
        self._frame_busy    = False           # ワーカースレッド動作中フラグ
        self._cap_lock      = threading.Lock()  # cap.set/read を保護するロック

        # ── UI ────────────────────────────────────
        self.frame = ttk.LabelFrame(parent, text=f'cam{cam_idx + 1}', padding=4)

        self.canvas = tk.Canvas(
            self.frame, width=PANEL_W, height=PANEL_H, bg='#1a1a1a',
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, columnspan=7, sticky='nsew')
        self._show_placeholder()

        self.frame_label = ttk.Label(self.frame, text='-- / --', anchor='center')
        self.frame_label.grid(row=1, column=0, columnspan=7, sticky='ew', pady=(2, 0))

        # ナビゲーション（◀◀=-1000  ◀=-100  ◀=-1  [entry]  ▶=+1  ▶=+100  ▶▶=+1000）
        ttk.Button(self.frame, text='◀◀', width=3,
                   command=lambda: self.seek_rel(-1000)).grid(row=2, column=0)
        ttk.Button(self.frame, text='◀',  width=3,
                   command=lambda: self.seek_rel(-100)).grid(row=2, column=1)
        ttk.Button(self.frame, text='◀',  width=3,
                   command=lambda: self.seek_rel(-1)).grid(row=2, column=2)
        self.frame_entry = ttk.Entry(self.frame, width=7, justify='center')
        self.frame_entry.grid(row=2, column=3)
        self.frame_entry.bind('<Return>', self._on_entry)
        ttk.Button(self.frame, text='▶',  width=3,
                   command=lambda: self.seek_rel(1)).grid(row=2, column=4)
        ttk.Button(self.frame, text='▶',  width=3,
                   command=lambda: self.seek_rel(100)).grid(row=2, column=5)
        ttk.Button(self.frame, text='▶▶', width=3,
                   command=lambda: self.seek_rel(1000)).grid(row=2, column=6)

        # Sync frame + Load ボタンを一行に
        ef_row = ttk.Frame(self.frame)
        ef_row.grid(row=3, column=0, columnspan=7, sticky='ew', pady=(3, 0))
        ttk.Label(ef_row, text='Sync:').pack(side='left')
        ttk.Entry(ef_row, textvariable=self.event_frame, width=7,
                  justify='center').pack(side='left', padx=2)
        ttk.Button(ef_row, text='←現在', width=5,
                   command=self._set_event_frame).pack(side='left')
        ttk.Separator(ef_row, orient='vertical').pack(side='left', fill='y', padx=4)
        ttk.Button(ef_row, text=f'📂', width=3,
                   command=self.load_video).pack(side='left')
        ttk.Button(ef_row, text='+M', width=3,
                   command=self.load_multi).pack(side='left', padx=(2, 0))

    # ── 内部ヘルパー ──────────────────────────────
    def _show_placeholder(self):
        self.canvas.delete('all')
        cw = int(self.canvas.cget('width'))
        ch = int(self.canvas.cget('height'))
        self.canvas.create_text(
            cw // 2, ch // 2,
            text=f'cam{self.cam_idx + 1}\n(no video)',
            fill='#555', font=('Arial', 13), justify='center',
        )

    def _on_entry(self, _=None):
        try:
            self.show_frame(int(self.frame_entry.get()))
        except ValueError:
            pass

    def _set_event_frame(self):
        self.event_frame.set(self.current_frame)

    def _release_all(self):
        self._frame_req = None   # ワーカーに停止を通知
        with self._cap_lock:     # 現在の read が終わってから解放
            for c in self.caps:
                c.release()
            self.caps.clear()
            self.paths.clear()
            self.segment_frames.clear()
            self.cum_frames = [0]
            self.total_frames = 0
            self.current_frame = 0

    def _resolve(self, global_idx: int) -> tuple[cv2.VideoCapture, int]:
        global_idx = max(0, min(global_idx, self.total_frames - 1))
        for seg in range(len(self.caps) - 1, -1, -1):
            if global_idx >= self.cum_frames[seg]:
                return self.caps[seg], global_idx - self.cum_frames[seg]
        return self.caps[0], global_idx

    def _finish_load(self):
        self.cum_frames = [0]
        for n in self.segment_frames:
            self.cum_frames.append(self.cum_frames[-1] + n)
        self.total_frames = self.cum_frames[-1]
        n = len(self.paths)
        label = (f'cam{self.cam_idx + 1}: {Path(self.paths[0]).name}'
                 if n == 1
                 else f'cam{self.cam_idx + 1}: [{n} files] {Path(self.paths[0]).name} …')
        self.frame.config(text=label)
        self.show_frame(0)

    # ── 公開 API ─────────────────────────────────
    def load_video(self, paths: list[str] | str | None = None):
        if paths is None:
            p = filedialog.askopenfilename(
                title=f'Select cam{self.cam_idx + 1} video',
                filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv'), ('All', '*.*')],
            )
            if not p:
                return
            paths = [p]
        elif isinstance(paths, str):
            paths = [paths]

        self._release_all()
        for p in sorted(paths):
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                messagebox.showerror('Error', f'Cannot open: {p}')
                cap.release()
                continue
            self.caps.append(cap)
            self.paths.append(p)
            self.segment_frames.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if self.caps:
            self._finish_load()

    def load_multi(self):
        ps = filedialog.askopenfilenames(
            title=f'cam{self.cam_idx + 1}: 分割ファイルをすべて選択',
            filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv'), ('All', '*.*')],
        )
        if ps:
            self.load_video(list(ps))

    def show_frame(self, global_idx: int):
        """フレーム表示をリクエストする（実際の読み込みはバックグラウンドで行う）。"""
        if not self.caps:
            return
        global_idx = max(0, min(global_idx, self.total_frames - 1))
        self.current_frame = global_idx
        self._update_labels(global_idx)
        # 最新リクエストを更新してワーカー起動
        self._frame_req = global_idx
        if not self._frame_busy:
            self._frame_busy = True
            threading.Thread(target=self._load_worker, daemon=True).start()

    def _update_labels(self, global_idx: int):
        seg_idx = next(
            (i - 1 for i, c in enumerate(self.cum_frames[1:], 1) if global_idx < c),
            len(self.caps) - 1,
        )
        seg_info = f'  [{seg_idx + 1}/{len(self.caps)}]' if len(self.caps) > 1 else ''
        self.frame_label.config(
            text=f'{global_idx}  /  {self.total_frames - 1}{seg_info}')
        self.frame_entry.delete(0, 'end')
        self.frame_entry.insert(0, str(global_idx))

    def _load_worker(self):
        """バックグラウンドでフレームを読み込む。連続リクエスト時は最新のみ処理。"""
        while True:
            target = self._frame_req
            if target is None:       # _release_all() に通知された
                self._frame_busy = False
                return
            cw = int(self.canvas.cget('width'))
            ch = int(self.canvas.cget('height'))

            ret, img = False, None
            with self._cap_lock:
                if not self.caps:    # リリース済みなら終了
                    self._frame_busy = False
                    return
                cap, local_idx = self._resolve(target)
                cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
                ret, img = cap.read()

            pil_img = None
            if ret:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
                scale = min(cw / w, ch / h)
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                img_rgb = cv2.resize(img_rgb, (nw, nh))
                pil_img = Image.fromarray(img_rgb)

            if self._frame_req == target:
                # リクエストが変わっていなければ表示して終了
                if pil_img is not None:
                    self.canvas.after(
                        0, lambda p=pil_img, w=cw, h=ch: self._display_frame(p, w, h))
                self._frame_busy = False
                return
            # リクエストが更新された → 最新フレームを読み込み直す（中間フレームはスキップ）

    def _display_frame(self, pil_img: Image.Image, cw: int, ch: int):
        """メインスレッドでキャンバスに描画する。"""
        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete('all')
        self.canvas.create_image(cw // 2, ch // 2, anchor='center', image=self._photo)

    def seek_rel(self, delta: int):
        self.show_frame(self.current_frame + delta)

    # ── 状態 I/O ─────────────────────────────────
    def get_state(self) -> dict:
        return {
            'paths':         self.paths,
            'event_frame':   self.event_frame.get(),
            'total_frames':  self.total_frames,
            'current_frame': self.current_frame,
        }

    def restore_state(self, state: dict):
        paths = state.get('paths') or []
        if not paths and state.get('path'):
            paths = [state['path']]
        if paths:
            self.load_video(paths)
        self.event_frame.set(state.get('event_frame', 0))
        if self.caps:
            self.show_frame(state.get('current_frame', 0))


# ══════════════════════════════════════════════════════
class SyncApp:
    def __init__(self, root: tk.Tk):
        self.root             = root
        self.root.title('6カメラ動画同期ツール')
        self.sync_offsets     = [0] * N_CAMS
        self.synced           = False
        self.sync_pos         = 0
        self._json_path       = 'sync_config.json'
        self.calib_wand_csv   = tk.StringVar(value='wand_annotations.csv')
        self.calib_start_frame = tk.IntVar(value=0)
        self.calib_n_frames    = tk.IntVar(value=30)    # キャリブ用スキャンフレーム数
        self.calib_max_frames  = tk.IntVar(value=500)
        self._log_queue: queue.Queue | None = None
        self._calib_running   = False
        self._slider_updating = False
        self.single_view_active = False
        self.cam_select_var   = tk.StringVar(value='cam1')

        # ── Step 4: Pose 3D 関連変数 ──────────────
        self.pose_cam_enabled = [tk.BooleanVar(value=(i != 4)) for i in range(N_CAMS)]  # cam5 off
        self.pose_calib_npz   = tk.StringVar(value='camera_calibration.npz')
        self.pose_start_frame = tk.IntVar(value=0)
        self.pose_end_frame   = tk.IntVar(value=1000)
        self.pose_vis_thresh  = tk.DoubleVar(value=0.5)
        self.pose_det_conf    = tk.DoubleVar(value=0.8)
        self.pose_rerun_thresh = tk.DoubleVar(value=0.5)
        self.pose_preview_cam_var = tk.StringVar(value='cam1')
        # Step 5: Stick Check 用のカメラ別 visibility 閾値
        self.stick_thresh_vars = {
            f'cam{i+1}': tk.DoubleVar(value=0.5) for i in range(N_CAMS)
        }
        self._pose_running    = False
        self._pose_log_queue: queue.Queue | None = None
        self._recon_log_queue: queue.Queue | None = None
        self._pose_csv_data: dict = {}   # {cam_name: pd.DataFrame}
        self._pose_skel_photo = None
        self._pose_skel_smooth_photo = None
        self.in_pose_preview_mode = False
        # 補間・平滑化パラメータ
        self.smooth_max_gap_var = tk.IntVar(value=5)   # 補間最大連続欠損フレーム数
        self.smooth_window_var  = tk.IntVar(value=7)   # 平滑化ウィンドウ幅（奇数推奨）
        self._pose_smooth_cache: dict = {}              # (cam, gap, win) → smoothed df
        self._resize_after_id = None                    # debounce ID for panel resize
        # 3D Recon データ選択
        self.recon_data_mode = tk.StringVar(value='raw')   # 'raw' or 'smooth'

        # ── Wand Annotation 関連変数 ──────────────
        self.wand_pose_names    = ['pose1', 'pose2', 'pose3', 'pose4', 'pose5']
        self.wand_point_labels  = ['0.0m', '0.5m', '1.0m', '1.5m']
        self.wand_cam_var       = tk.StringVar(value='cam1')
        self.wand_pose_idx      = 0
        self.wand_click_mode    = False
        self.wand_clicked_pts   = []   # [(u, v), ...] 画像ピクセル座標
        self._wand_cap          = None
        self._wand_cap_path     = ''
        self._wand_frame_idx    = 0
        self._wand_total_frames = 0
        self._wand_photo        = None
        self._wand_img_scale    = 1.0
        self._wand_img_offset   = (0, 0)
        self._wand_img_size     = (1920, 1080)
        self.wand_annotations   = {}   # (cam, pose, label) → {'frame', 'u', 'v'}
        self._wand_active       = False

        self._build_ui()
        self._bind_keys()

    # ── UI 構築 ───────────────────────────────────
    def _build_ui(self):
        # ── パネルエリア（固定高さ）──────────────
        self.panel_area = ttk.Frame(self.root, padding=6)
        self.panel_area.pack(fill='x')

        self.panels: list[VideoPanel] = []
        for i in range(N_CAMS):
            row, col = divmod(i, COLS)
            p = VideoPanel(self.panel_area, i)
            p.frame.grid(row=row, column=col, padx=3, pady=3, sticky='nsew')
            self.panels.append(p)
        for c in range(COLS):
            self.panel_area.columnconfigure(c, weight=1)

        # Step 4 用スティックフィギュアキャンバス（panel_area 右半分に配置、初期は非表示）
        self._skel_lf = ttk.LabelFrame(self.panel_area, text='Stick Figure (Raw)', padding=2)
        self._skel_canvas = tk.Canvas(
            self._skel_lf, width=PANEL_W, height=PANEL_H,
            bg='white', highlightthickness=0,
        )
        self._skel_canvas.pack(fill='both', expand=True)

        # 補間・平滑化後スティックフィギュアキャンバス（Stick Check 時のみ表示）
        self._skel_smooth_lf = ttk.LabelFrame(
            self.panel_area, text='Stick Figure (Interp + Smooth)', padding=2)
        self._skel_smooth_canvas = tk.Canvas(
            self._skel_smooth_lf, width=PANEL_W, height=PANEL_H,
            bg='#f0f8ff', highlightthickness=0,
        )
        self._skel_smooth_canvas.pack(fill='both', expand=True)

        # Wand Annotation 用キャンバス（panel_area 全面に配置、初期は非表示）
        self._wand_panel_lf = ttk.LabelFrame(self.panel_area, text='Wand Annotation', padding=2)
        self._wand_panel_canvas = tk.Canvas(
            self._wand_panel_lf, bg='#1a1a1a', cursor='crosshair',
            highlightthickness=0,
        )
        self._wand_panel_canvas.pack(fill='both', expand=True)
        self._wand_panel_canvas.bind('<Button-1>', self._wand_on_canvas_click)
        self._wand_panel_canvas.bind(
            '<Configure>',
            lambda e: self._wand_show_frame(self._wand_frame_idx)
            if self._wand_active and self._wand_cap else None,
        )

        # ── 共通同期ナビバー ──────────────────────
        nav_bar = ttk.Frame(self.root, padding=(8, 2))
        nav_bar.pack(fill='x')

        # ボタン群（左端）
        ttk.Label(nav_bar, text='Synced:').pack(side='left')
        ttk.Button(nav_bar, text='|◀', width=3,
                   command=lambda: self._sync_all_to(0)).pack(side='left', padx=1)
        ttk.Button(nav_bar, text='◀◀', width=3,
                   command=lambda: self.sync_seek(-1000)).pack(side='left', padx=1)
        ttk.Button(nav_bar, text='◀',  width=3,
                   command=lambda: self.sync_seek(-1)).pack(side='left', padx=1)
        ttk.Button(nav_bar, text='▶',  width=3,
                   command=lambda: self.sync_seek(1)).pack(side='left', padx=1)
        ttk.Button(nav_bar, text='▶▶', width=3,
                   command=lambda: self.sync_seek(1000)).pack(side='left', padx=1)

        # フレーム番号入力（直接ジャンプ）
        self.sync_nav_var = tk.IntVar(value=0)
        sync_entry = ttk.Entry(nav_bar, textvariable=self.sync_nav_var,
                               width=7, justify='center')
        sync_entry.pack(side='left', padx=(6, 2))
        sync_entry.bind('<Return>',
                        lambda e: self.synced and self._sync_all_to(self.sync_nav_var.get()))

        self.sync_pos_label = ttk.Label(nav_bar, text='/ --', width=8)
        self.sync_pos_label.pack(side='left')

        # スライダー（残りの横幅を使用）
        self.sync_slider = ttk.Scale(
            nav_bar, from_=0, to=1000, orient='horizontal',
            command=self._on_slider,
        )
        self.sync_slider.pack(side='left', fill='x', expand=True, padx=8)

        # カメラ選択ドロップダウン＋ビュー切替
        ttk.Separator(nav_bar, orient='vertical').pack(side='left', fill='y', padx=6)
        cam_combo = ttk.Combobox(
            nav_bar, textvariable=self.cam_select_var,
            values=[f'cam{i + 1}' for i in range(N_CAMS)],
            width=6, state='readonly',
        )
        cam_combo.pack(side='left', padx=2)
        cam_combo.bind('<<ComboboxSelected>>', self._on_cam_select)
        self.view_btn = ttk.Button(
            nav_bar, text='⊡ 1cam', width=8, command=self._toggle_view)
        self.view_btn.pack(side='left', padx=2)

        # ステータス（右端）
        self.status_var = tk.StringVar(value='Ready')
        ttk.Label(nav_bar, textvariable=self.status_var,
                  foreground='gray', wraplength=220).pack(side='left', padx=4)

        # ── Notebook（残りのスペースを使う）──────
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=6, pady=(0, 6))
        self._build_sync_tab()          # Step 1: Sync
        self._build_wand_tab()          # Step 2: Wand Annotation
        self._build_calib_tab()         # Step 3: Calibration
        self._build_results_tab()       # Step 4: Results
        self._build_pose3d_tab()        # Step 5: Pose Recognition
        self._build_stick_check_tab()   # Step 6: Stick Check（閾値確認）
        self._build_pose5_tab()         # Step 7: 3D Recon
        self.nb.bind('<<NotebookTabChanged>>', self._on_tab_change)

    def _build_sync_tab(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text='  Step 1: Sync  ')
        ttk.Button(tab, text='📂 Load All',
                   command=self.load_all).pack(side='left', padx=3)
        ttk.Button(tab, text='💾 Save JSON',
                   command=self.save_json).pack(side='left', padx=3)
        ttk.Button(tab, text='📄 Load JSON',
                   command=self.load_json).pack(side='left', padx=3)
        ttk.Separator(tab, orient='vertical').pack(side='left', fill='y', padx=10)
        ttk.Button(tab, text='🔗  Sync →',
                   command=self.do_sync).pack(side='left', padx=3)
        ttk.Label(tab, text='各カメラの Sync frame を設定して Sync を押してください',
                  foreground='gray').pack(side='left', padx=8)
        # アクティブカメラチェックボックス
        ttk.Separator(tab, orient='vertical').pack(side='left', fill='y', padx=10)
        ttk.Label(tab, text='Active cams:').pack(side='left', padx=(0, 4))
        for i in range(N_CAMS):
            ttk.Checkbutton(tab, text=f'{i + 1}',
                            variable=self.pose_cam_enabled[i]).pack(side='left')

    def _build_calib_tab(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text='  Step 3: Calibration  ')

        # Wand CSV 選択（任意）
        csv_row = ttk.Frame(tab)
        csv_row.pack(fill='x', pady=(0, 4))
        ttk.Label(csv_row, text='Wand CSV (Optional):').pack(side='left')
        ttk.Entry(csv_row, textvariable=self.calib_wand_csv,
                  width=38).pack(side='left', padx=4)
        ttk.Button(csv_row, text='📂 Browse',
                   command=self._browse_wand_csv).pack(side='left')
        ttk.Label(csv_row, text='（なくても可）', foreground='gray').pack(side='left', padx=4)

        # Start frame + 推定フレーム数 + BA最大フレーム数
        param_row = ttk.Frame(tab)
        param_row.pack(fill='x', pady=(0, 6))
        ttk.Label(param_row, text='Start frame (synced):').pack(side='left')
        ttk.Entry(param_row, textvariable=self.calib_start_frame,
                  width=8, justify='center').pack(side='left', padx=4)
        ttk.Button(param_row, text='← 現在位置',
                   command=lambda: self.calib_start_frame.set(self.sync_pos)
                   ).pack(side='left', padx=(0, 12))
        ttk.Label(param_row, text='推定フレーム数:').pack(side='left')
        ttk.Entry(param_row, textvariable=self.calib_n_frames,
                  width=6, justify='center').pack(side='left', padx=4)
        ttk.Label(param_row, text='BA最大:').pack(side='left', padx=(8, 0))
        ttk.Entry(param_row, textvariable=self.calib_max_frames,
                  width=6, justify='center').pack(side='left', padx=4)
        ttk.Label(param_row,
                  text='(Run Cal でN フレームをスキャン → 人検出 & v≥0.95 のフレームのみでキャリブ)',
                  foreground='gray').pack(side='left', padx=4)

        # Run ボタン
        run_row = ttk.Frame(tab)
        run_row.pack(fill='x', pady=(0, 6))
        self.run_calib_btn = ttk.Button(
            run_row, text='▶  Run Calibration', command=self.run_calibration)
        self.run_calib_btn.pack(side='left', padx=(0, 8))
        self.calib_progress = tk.StringVar(value='')
        ttk.Label(run_row, textvariable=self.calib_progress,
                  foreground='#0088cc').pack(side='left')

        # ログ出力
        log_frame = ttk.LabelFrame(tab, text='Output', padding=4)
        log_frame.pack(fill='both', expand=True)
        self.calib_log = tk.Text(
            log_frame, wrap='none', font=('Courier', 9),
            bg='#1e1e1e', fg='#d4d4d4', state='disabled',
        )
        sb_y = ttk.Scrollbar(log_frame, orient='vertical',
                             command=self.calib_log.yview)
        sb_x = ttk.Scrollbar(log_frame, orient='horizontal',
                             command=self.calib_log.xview)
        self.calib_log.configure(yscrollcommand=sb_y.set,
                                 xscrollcommand=sb_x.set)
        sb_y.pack(side='right', fill='y')
        sb_x.pack(side='bottom', fill='x')
        self.calib_log.pack(fill='both', expand=True)

    def _build_results_tab(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text='  Step 4: Results  ')

        # ツールバー
        tb = ttk.Frame(tab)
        tb.pack(fill='x', pady=(0, 6))
        ttk.Button(tb, text='🔄  Load Results',
                   command=self._load_results).pack(side='left', padx=3)
        self.results_status = tk.StringVar(
            value='キャリブレーション完了後に "Load Results" を押してください')
        ttk.Label(tb, textvariable=self.results_status,
                  foreground='gray').pack(side='left', padx=8)

        # 左右に分割（図 ← → 誤差テーブル）
        body = ttk.Frame(tab)
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # 左: XZ 平面カメラ配置
        fig_frame = ttk.LabelFrame(body, text='Camera Positions (XZ plane)', padding=4)
        fig_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 4))
        self._results_fig = Figure(figsize=(5, 3.5), tight_layout=True)
        self._results_canvas = FigureCanvasTkAgg(self._results_fig, master=fig_frame)
        self._results_canvas.get_tk_widget().pack(fill='both', expand=True)

        # 右: 再投影誤差テーブル
        err_frame = ttk.LabelFrame(body, text='Reprojection Error [px]', padding=4)
        err_frame.grid(row=0, column=1, sticky='nsew')
        cols = ('Camera', 'N obs', 'Mean', 'Median', 'RMS', 'Max')
        self._err_tree = ttk.Treeview(
            err_frame, columns=cols, show='headings', height=8)
        for c, w in zip(cols, (60, 60, 60, 60, 60, 60)):
            self._err_tree.heading(c, text=c)
            self._err_tree.column(c, width=w, anchor='center')
        sb = ttk.Scrollbar(err_frame, orient='vertical', command=self._err_tree.yview)
        self._err_tree.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self._err_tree.pack(fill='both', expand=True)

    # ── キーバインド ──────────────────────────────
    def _bind_keys(self):
        self.root.bind('<Left>',        lambda e: self.sync_seek(-1))
        self.root.bind('<Right>',       lambda e: self.sync_seek(1))
        self.root.bind('<Shift-Left>',  lambda e: self.sync_seek(-1000))
        self.root.bind('<Shift-Right>', lambda e: self.sync_seek(1000))

    # ── 同期ナビ ──────────────────────────────────
    def _on_slider(self, val):
        if self._slider_updating:
            return
        pos = int(float(val))
        if self.synced:
            self._sync_all_to(pos)

    def _sync_all_to(self, pos: int):
        # スライダーの範囲に収める
        try:
            to_val = int(self.sync_slider.cget('to'))
        except Exception:
            to_val = 0
        pos = max(0, min(pos, to_val) if to_val > 0 else pos)
        self.sync_pos = pos
        for i, p in enumerate(self.panels):
            if p.caps:
                p.show_frame(self.sync_offsets[i] + pos)
        self.sync_pos_label.config(text=f'/ {to_val}')
        # スライダーとエントリを更新（再帰防止）
        self._slider_updating = True
        self.sync_slider.set(pos)
        self.sync_nav_var.set(pos)
        self._slider_updating = False
        # Step 4 ポーズプレビューモード時はスティックフィギュアも更新
        if self.in_pose_preview_mode:
            self._update_skel_canvas(pos)
            self._update_smooth_skel_canvas(pos)

    def _update_slider_range(self):
        """動画がロードされているカメラから最大同期ポジションを算出してスライダー範囲を設定"""
        max_pos = 0
        for i, p in enumerate(self.panels):
            if p.caps:
                avail = p.total_frames - 1 - self.sync_offsets[i]
                max_pos = max(max_pos, avail)
        self.sync_slider.config(to=max(1, max_pos))
        self.sync_pos_label.config(text=f'/ {max(1, max_pos)}')

    def sync_seek(self, delta: int):
        if self.synced:
            self._sync_all_to(self.sync_pos + delta)

    # ── ビュー切替 ────────────────────────────────
    def _get_cam_idx(self) -> int:
        try:
            return int(self.cam_select_var.get().replace('cam', '')) - 1
        except ValueError:
            return 0

    def _on_cam_select(self, _=None):
        if self.in_pose_preview_mode:
            self._enter_pose_preview_mode()
        elif self.single_view_active:
            self._enter_single_view(self._get_cam_idx())

    def _toggle_view(self):
        if self.in_pose_preview_mode:
            self._exit_pose_preview_mode(restore_view=False)
            self._enter_multi_view()
        elif self.single_view_active:
            self._enter_multi_view()
        else:
            self._enter_single_view(self._get_cam_idx())

    def _enter_single_view(self, cam_idx: int):
        self.single_view_active = True
        self.cam_select_var.set(f'cam{cam_idx + 1}')
        self.view_btn.config(text='⊞ All cams')

        # パネルエリアを縦方向にも拡張
        self.panel_area.pack_configure(fill='both', expand=True)
        self.nb.pack_configure(fill='x', expand=False)

        # 選択カメラだけ表示（3列すべて使用）
        for i, p in enumerate(self.panels):
            if i == cam_idx:
                p.frame.grid(row=0, column=0, columnspan=COLS,
                             padx=4, pady=4, sticky='nsew')
            else:
                p.frame.grid_remove()

        self.panel_area.grid_rowconfigure(0, weight=1)
        self.panel_area.grid_rowconfigure(1, weight=0)
        for c in range(COLS):
            self.panel_area.grid_columnconfigure(c, weight=1)

        # キャンバスをパネルエリアに合わせてリサイズ
        self.root.update_idletasks()
        self._resize_single_canvas(cam_idx)

    def _resize_single_canvas(self, cam_idx: int):
        p = self.panels[cam_idx]
        aw = max(PANEL_W, self.panel_area.winfo_width() - 14)
        ah = max(PANEL_H, self.panel_area.winfo_height() - 92)  # label+nav+sync_row
        p.canvas.config(width=aw, height=ah)
        if p.caps:
            p.show_frame(p.current_frame)
        else:
            p._show_placeholder()

    def _enter_multi_view(self):
        self.single_view_active = False
        self.view_btn.config(text='⊡ 1cam')

        # パネルエリアを固定高さに戻す
        self.panel_area.pack_configure(fill='x', expand=False)
        self.nb.pack_configure(fill='both', expand=True)

        for c in range(COLS):
            self.panel_area.grid_columnconfigure(c, weight=1)
        for r in range(2):
            self.panel_area.grid_rowconfigure(r, weight=0)

        # 全パネルを元のサイズ・位置に戻す
        for i, p in enumerate(self.panels):
            row, col = divmod(i, COLS)
            p.canvas.config(width=PANEL_W, height=PANEL_H)
            p.frame.grid(row=row, column=col, padx=3, pady=3, sticky='nsew')
            if p.caps:
                p.show_frame(p.current_frame)
            else:
                p._show_placeholder()

    # ── Step 1 ────────────────────────────────────
    def load_all(self):
        paths = filedialog.askopenfilenames(
            title='Select 6 videos (cam1 → cam6 の順)',
            filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv'), ('All', '*.*')],
        )
        if not paths:
            return
        for i, path in enumerate(sorted(paths)[:N_CAMS]):
            self.panels[i].load_video([path])

    def do_sync(self):
        missing = [i + 1 for i in range(N_CAMS) if not self.panels[i].caps]
        if missing and not messagebox.askyesno(
            'Warning', f'cam{missing} に動画が未読み込みです。続行しますか？'
        ):
            return
        raw = [p.event_frame.get() for p in self.panels]
        loaded_raw = [o for i, o in enumerate(raw) if self.panels[i].caps]
        min_off = min(loaded_raw) if loaded_raw else 0
        self.sync_offsets = [o - min_off for o in raw]
        self.synced = True
        self.sync_pos = 0
        self._update_slider_range()
        self._sync_all_to(0)
        self.status_var.set(
            f'Synced  offsets: {self.sync_offsets}'
        )
        self._autosave()
        self._enter_single_view(0)
        self.nb.select(1)   # → Step 2: Wand Annotation

    # ── Step 3: Calibration ───────────────────────
    def _browse_wand_csv(self):
        p = filedialog.askopenfilename(
            title='Select wand_annotations.csv',
            filetypes=[('CSV', '*.csv'), ('All', '*.*')],
        )
        if p:
            self.calib_wand_csv.set(p)

    def run_calibration(self):
        if not self.synced:
            messagebox.showerror('Error', '先に Step 1 で Sync を行ってください。')
            return
        if self._calib_running:
            messagebox.showinfo('Info', 'キャリブレーション実行中です。')
            return

        # 動画が読み込まれているアクティブカメラを収集
        enabled_panels = [
            (i, self.panels[i])
            for i in range(N_CAMS)
            if self.pose_cam_enabled[i].get() and self.panels[i].caps
        ]
        if not enabled_panels:
            messagebox.showerror('Error', '動画が読み込まれているアクティブカメラがありません。')
            return

        # キャリブ用スキャン範囲を決定（連続スキャン → 検出フレームのみ BA で使用）
        start_s  = self.calib_start_frame.get()
        n_frames = self.calib_n_frames.get()
        max_avail = 0
        for cam_i, panel in enabled_panels:
            avail = panel.total_frames - 1 - self.sync_offsets[cam_i] - start_s
            max_avail = max(max_avail, avail)
        if max_avail <= 0:
            messagebox.showerror('Error', 'スタートフレーム以降に利用可能なフレームがありません。')
            return
        actual_n = min(n_frames, max_avail)

        # sync_config.json にキャリブレーション設定を保存（マージ方式）
        try:
            with open(self._json_path, 'r', encoding='utf-8') as fh:
                existing_cfg = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_cfg = {}
        cfg = self._build_json()
        if 'cameras' in existing_cfg and 'cameras' in cfg:
            for old_c, new_c in zip(existing_cfg['cameras'], cfg['cameras']):
                if not new_c.get('paths') and old_c.get('paths'):
                    new_c['paths'] = old_c['paths']
                    new_c.setdefault('total_frames', old_c.get('total_frames', 0))
        old_calib = existing_cfg.get('calib', {})
        cfg['calib'] = {**old_calib, **cfg.get('calib', {})}
        # start_frames_per_cam: スキャン開始フレーム（検出済みフレームは est_camera_poses で絞込）
        cfg['calib']['start_frames_per_cam'] = {
            f'cam{i + 1}': start_s
            for i in range(N_CAMS)
            if self.panels[i].caps
        }
        existing_cfg.update(cfg)
        with open(self._json_path, 'w', encoding='utf-8') as fh:
            json.dump(existing_cfg, fh, indent=2, ensure_ascii=False)

        # ログクリア
        self.calib_log.config(state='normal')
        self.calib_log.delete('1.0', 'end')
        self.calib_log.config(state='disabled')

        self._calib_running = True
        self.run_calib_btn.config(state='disabled')
        self.calib_progress.set('ポーズ推定中…')
        self.status_var.set('Calibration: pose estimation…')

        self._log_queue = queue.Queue()

        def _worker():
            import json as _json
            total_cams = len(enabled_panels)

            # ── Step 1: 各カメラで連続範囲スキャン → 人検出フレームのみ保存 ──────────
            for step_i, (cam_i, panel) in enumerate(enabled_panels):
                cam_name = f'cam{cam_i + 1}'
                self._log_queue.put(
                    f'--- {cam_name} ポーズ推定 ({step_i + 1}/{total_cams}) ---\n'
                )

                # 連続範囲スキャン: start_s から actual_n フレームをカメラローカル座標に変換
                segments = []
                frames_remaining = actual_n
                for seg_i, path in enumerate(panel.paths):
                    if frames_remaining <= 0:
                        break
                    seg_start_abs = panel.cum_frames[seg_i]
                    cam_start_local = self.sync_offsets[cam_i] + start_s
                    cam_end_local   = self.sync_offsets[cam_i] + start_s + actual_n
                    local_start = max(0, cam_start_local - seg_start_abs)
                    local_end   = min(panel.segment_frames[seg_i], cam_end_local - seg_start_abs)
                    if local_end <= local_start:
                        continue
                    segments.append({'path': path, 'start': local_start, 'end': local_end})
                    frames_remaining -= (local_end - local_start)

                if not segments:
                    self._log_queue.put(f'  {cam_name}: 有効フレームなし、スキップ\n')
                    continue

                config = {
                    'cam_name':       cam_name,
                    'video_segments': segments,
                    'synced_start':   start_s,
                    'det_conf':       self.pose_det_conf.get(),
                    'pres_conf':      self.pose_det_conf.get(),
                    'track_conf':     self.pose_det_conf.get(),
                    'save_to':        'calib',   # calib.landmarks に保存（人未検出フレームは除去）
                }
                cfg_file = Path(f'_calib_{cam_name}_config.json')
                cfg_file.write_text(
                    _json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')

                try:
                    proc = subprocess.Popen(
                        [sys.executable, 'recog_mediapipe.py', '--config', str(cfg_file)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, encoding='utf-8', errors='replace',
                    )
                    for line in proc.stdout:
                        self._log_queue.put(strip_ansi(line))
                    proc.wait()
                    cfg_file.unlink(missing_ok=True)
                    if proc.returncode != 0:
                        self._log_queue.put(
                            f'  {cam_name}: ポーズ推定失敗 (code={proc.returncode})\n')
                        self._log_queue.put(('__done__', proc.returncode))
                        return
                except Exception as ex:
                    self._log_queue.put(f'  ERROR: {ex}\n')
                    self._log_queue.put(('__done__', -1))
                    return

            # ── Step 2: Bundle Adjustment ────────────────────────────────
            self._log_queue.put(('__progress__', 'キャリブレーション中…'))
            self._log_queue.put('\n--- キャリブレーション (Bundle Adjustment) ---\n')
            try:
                proc = subprocess.Popen(
                    [sys.executable, 'est_camera_poses.py'],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, encoding='utf-8', errors='replace',
                )
                for line in proc.stdout:
                    self._log_queue.put(strip_ansi(line))
                proc.wait()
                self._log_queue.put(('__done__', proc.returncode))
            except Exception as ex:
                self._log_queue.put(f'ERROR: {ex}\n')
                self._log_queue.put(('__done__', -1))

        threading.Thread(target=_worker, daemon=True).start()
        self.root.after(100, self._poll_log)

    def _poll_log(self):
        try:
            while True:
                item = self._log_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == '__done__':
                    self._on_calibration_complete(item[1])
                    return
                if isinstance(item, tuple) and item[0] == '__progress__':
                    self.calib_progress.set(item[1])
                    continue
                # text line
                self.calib_log.config(state='normal')
                self.calib_log.insert('end', item)
                self.calib_log.see('end')
                self.calib_log.config(state='disabled')
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _on_calibration_complete(self, returncode: int):
        self._calib_running = False
        self.run_calib_btn.config(state='normal')
        if returncode == 0:
            self.calib_progress.set('✓ 完了')
            self.status_var.set('Calibration complete.')
            self._load_results()
            self.nb.select(3)   # → Step 4: Results
        else:
            self.calib_progress.set(f'✗ 終了コード {returncode}')
            self.status_var.set(f'Calibration failed (code={returncode})')

    # ── Step 4: Results ──────────────────────────
    def _load_results(self):
        npz_path = Path('camera_calibration.npz')
        if not npz_path.exists():
            messagebox.showerror('Error',
                                 'camera_calibration.npz が見つかりません。\n'
                                 'キャリブレーションを先に実行してください。')
            return

        data = np.load(npz_path, allow_pickle=True)
        rvecs  = data['rvecs']
        tvecs  = data['tvecs']
        cam_names = list(data['camera_names'])

        cam_positions = []
        for i in range(len(rvecs)):
            R, _ = cv2.Rodrigues(rvecs[i])
            cam_positions.append(-R.T @ tvecs[i])
        positions = np.array(cam_positions)

        # matplotlib カメラ配置（XZ 平面のみ）
        self._results_fig.clear()
        colors = [f'C{i}' for i in range(len(cam_names))]
        ax = self._results_fig.add_subplot(111)

        for i, (pos, name) in enumerate(zip(positions, cam_names)):
            ax.scatter(pos[0], pos[2], color=colors[i], s=80, zorder=3)
            ax.annotate(name, (pos[0], pos[2]),
                        xytext=(5, 4), textcoords='offset points',
                        fontsize=8, color=colors[i], fontweight='bold')
        ax.scatter(0, 0, marker='+', s=150, color='black', zorder=4, linewidths=2)
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Z', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.35)

        self._results_canvas.draw()

        # 再投影誤差テーブル（sync_config.json の reprojection_errors キーから読む）
        for row in self._err_tree.get_children():
            self._err_tree.delete(row)

        err_data = None
        cfg_path = Path(self._json_path)
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
                err_data = cfg.get('reprojection_errors')
            except Exception:
                err_data = None

        if err_data is not None:
            for cam in cam_names:
                e = err_data.get(cam, {})
                if e.get('n_obs', 0) == 0:
                    self._err_tree.insert('', 'end', values=(cam, '0', '-', '-', '-', '-'))
                else:
                    self._err_tree.insert('', 'end', values=(
                        cam,
                        e.get('n_obs', '-'),
                        f"{e.get('mean', 0):.2f}",
                        f"{e.get('median', 0):.2f}",
                        f"{e.get('rms', 0):.2f}",
                        f"{e.get('max', 0):.2f}",
                    ))
        else:
            for cam in cam_names:
                self._err_tree.insert('', 'end',
                                      values=(cam, '(reprojection_errors なし)', '', '', '', ''))

        self.results_status.set(f'Loaded: {npz_path.resolve()}')

    # ── JSON I/O ──────────────────────────────────
    def _build_json(self) -> dict:
        return {
            'cameras':      [p.get_state() for p in self.panels],
            'sync_offsets': self.sync_offsets,
            'synced':       self.synced,
            'sync_pos':     self.sync_pos,
            'calib': {
                'wand_csv':         self.calib_wand_csv.get(),
                'start_frame':      self.calib_start_frame.get(),
                'max_calib_frames': self.calib_max_frames.get(),
            },
            'active_cameras': [
                f'cam{i + 1}' for i in range(N_CAMS)
                if self.pose_cam_enabled[i].get()
            ],
            'pose3d': {
                'npz_path':      self.pose_calib_npz.get(),
                'start_frame':   self.pose_start_frame.get(),
                'end_frame':     self.pose_end_frame.get(),
                'vis_thresh':    self.pose_vis_thresh.get(),
                'det_conf':      self.pose_det_conf.get(),
                'cam_thresholds': {
                    cam: var.get()
                    for cam, var in self.stick_thresh_vars.items()
                },
            },
        }

    def _autosave(self):
        # 既存ファイルを読んでマージし、外部スクリプトが書いたキーを保持する
        try:
            with open(self._json_path, 'r', encoding='utf-8') as fh:
                existing = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {}

        new_data = self._build_json()

        # cameras: 動画が未ロードのパネルは既存パスを保持する
        if 'cameras' in existing and 'cameras' in new_data:
            for old_cam, new_cam in zip(existing['cameras'], new_data['cameras']):
                if not new_cam.get('paths') and old_cam.get('paths'):
                    new_cam['paths'] = old_cam['paths']
                    new_cam.setdefault('total_frames', old_cam.get('total_frames', 0))

        # calib: est_camera_poses.py が書いた npz_path 等を保持しつつ新規値を上書き
        old_calib = existing.get('calib', {})
        new_data['calib'] = {**old_calib, **new_data.get('calib', {})}

        # pose3d: recog_mediapipe.py が書いた csv_paths を保持しつつ新規値を上書き
        old_pose3d = existing.get('pose3d', {})
        new_data['pose3d'] = {**old_pose3d, **new_data.get('pose3d', {})}

        existing.update(new_data)

        with open(self._json_path, 'w', encoding='utf-8') as fh:
            json.dump(existing, fh, indent=2, ensure_ascii=False)

    def save_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON', '*.json')],
            initialfile=Path(self._json_path).name,
        )
        if not path:
            return
        self._json_path = path
        self._autosave()
        self.status_var.set(f'Saved → {path}')

    def load_json(self):
        path = filedialog.askopenfilename(
            filetypes=[('JSON', '*.json')],
            initialfile='sync_config.json',
        )
        if not path:
            return
        self._json_path = path
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        for i, state in enumerate(data.get('cameras', [])):
            if i < N_CAMS:
                self.panels[i].restore_state(state)
        self.sync_offsets = data.get('sync_offsets', [0] * N_CAMS)
        self.synced       = data.get('synced', False)
        self.sync_pos     = data.get('sync_pos', 0)
        if self.synced:
            self._update_slider_range()
            self._sync_all_to(self.sync_pos)
            self._enter_single_view(0)
        calib = data.get('calib', {})
        if calib.get('wand_csv'):
            self.calib_wand_csv.set(calib['wand_csv'])
        if 'start_frame' in calib:
            self.calib_start_frame.set(calib['start_frame'])
        if calib.get('max_calib_frames'):
            self.calib_max_frames.set(calib['max_calib_frames'])

        # active_cameras
        active = data.get('active_cameras', [])
        if active:
            for i in range(N_CAMS):
                self.pose_cam_enabled[i].set(f'cam{i + 1}' in active)

        # pose3d
        pose3d = data.get('pose3d', {})
        # npz_path は pose3d か calib どちらにあっても読む
        _npz = pose3d.get('npz_path') or calib.get('npz_path')
        if _npz:
            self.pose_calib_npz.set(_npz)
        if 'start_frame' in pose3d:
            self.pose_start_frame.set(pose3d['start_frame'])
        if 'end_frame' in pose3d:
            self.pose_end_frame.set(pose3d['end_frame'])
        if 'vis_thresh' in pose3d:
            self.pose_vis_thresh.set(pose3d['vis_thresh'])
        if 'det_conf' in pose3d:
            self.pose_det_conf.set(pose3d['det_conf'])
        if 'cam_thresholds' in pose3d:
            for cam, val in pose3d['cam_thresholds'].items():
                if cam in self.stick_thresh_vars:
                    self.stick_thresh_vars[cam].set(val)

        self.status_var.set(f'Loaded: {Path(path).name}')


    # ── Step 4: Pose 3D ───────────────────────────

    def _build_pose3d_tab(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text='  Step 5: Pose Recognition  ')

        # 1. 設定フレーム
        settings_frame = ttk.LabelFrame(tab, text='Settings', padding=6)
        settings_frame.pack(fill='x', pady=(0, 6))

        row1 = ttk.Frame(settings_frame)
        row1.pack(fill='x', pady=(0, 4))
        ttk.Label(row1, text='Calib NPZ:').pack(side='left')
        ttk.Entry(row1, textvariable=self.pose_calib_npz,
                  width=35).pack(side='left', padx=4)
        ttk.Button(row1, text='Browse',
                   command=self._browse_calib_npz).pack(side='left')
        ttk.Separator(row1, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Label(row1, text='Det conf:').pack(side='left')
        ttk.Entry(row1, textvariable=self.pose_det_conf,
                  width=5, justify='center').pack(side='left', padx=4)
        ttk.Label(row1, text='  ※ 全ランドマークを visibility スコア付きで保存',
                  foreground='gray').pack(side='left', padx=(8, 0))

        row2 = ttk.Frame(settings_frame)
        row2.pack(fill='x')
        ttk.Label(row2, text='Start:').pack(side='left')
        ttk.Entry(row2, textvariable=self.pose_start_frame,
                  width=8, justify='center').pack(side='left', padx=4)
        ttk.Button(row2, text='←現在位置',
                   command=lambda: self.pose_start_frame.set(self.sync_pos)
                   ).pack(side='left', padx=(0, 12))
        ttk.Label(row2, text='End:').pack(side='left')
        ttk.Entry(row2, textvariable=self.pose_end_frame,
                  width=8, justify='center').pack(side='left', padx=4)
        ttk.Button(row2, text='←現在位置',
                   command=lambda: self.pose_end_frame.set(self.sync_pos)
                   ).pack(side='left')

        # 2. Run ボタン行
        run_row = ttk.Frame(tab)
        run_row.pack(fill='x', pady=(0, 4))
        self.run_mediapipe_btn = ttk.Button(
            run_row, text='▶ Run MediaPipe', command=self.run_mediapipe)
        self.run_mediapipe_btn.pack(side='left', padx=(0, 8))
        self.pose_progress_var = tk.StringVar(value='')
        ttk.Label(run_row, textvariable=self.pose_progress_var,
                  foreground='#0088cc').pack(side='left')

        # 3. ログ
        log_frame = ttk.LabelFrame(tab, text='Output', padding=4)
        log_frame.pack(fill='x', pady=(0, 6))
        self.pose_log = tk.Text(
            log_frame, wrap='none', font=('Courier', 9),
            bg='#1e1e1e', fg='#d4d4d4', state='disabled', height=20,
        )
        sb_y = ttk.Scrollbar(log_frame, orient='vertical', command=self.pose_log.yview)
        sb_x = ttk.Scrollbar(log_frame, orient='horizontal', command=self.pose_log.xview)
        self.pose_log.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.pack(side='right', fill='y')
        sb_x.pack(side='bottom', fill='x')
        self.pose_log.pack(fill='both', expand=True)

        # 4. 次ステップ案内
        ttk.Label(tab,
            text='認識完了後は Step 5 でスティックフィギュアを確認し、カメラ別閾値を設定してください。',
            foreground='gray').pack(anchor='w', pady=(4, 0))

    def _browse_calib_npz(self):
        p = filedialog.askopenfilename(
            title='Select camera_calibration.npz',
            filetypes=[('NumPy NPZ', '*.npz'), ('All', '*.*')],
        )
        if p:
            self.pose_calib_npz.set(p)

    # ── Pose 3D ログ書き込みヘルパー ──────────────

    def _pose_log_write(self, text: str):
        """メインスレッドからポーズログに書き込む。"""
        self.pose_log.config(state='normal')
        self.pose_log.insert('end', text)
        self.pose_log.see('end')
        self.pose_log.config(state='disabled')

    def _poll_pose_log(self):
        """ポーズログキューをポーリングしてウィジェットに反映する。"""
        if self._pose_log_queue is None:
            return
        try:
            while True:
                item = self._pose_log_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == '__done__':
                    self._on_pose_step_done(item[1], item[2])
                    return
                self._pose_log_write(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_pose_log)

    # ── MediaPipe 実行 ──────────────────────────────

    def run_mediapipe(self):
        if self._pose_running:
            messagebox.showinfo('Info', 'すでに実行中です。')
            return
        if not self.synced:
            messagebox.showerror('Error', '先に Step 1 で Sync を行ってください。')
            return

        # sync_config.json を更新
        self._autosave()

        # ログクリア
        self.pose_log.config(state='normal')
        self.pose_log.delete('1.0', 'end')
        self.pose_log.config(state='disabled')

        self._pose_running = True
        self.run_mediapipe_btn.config(state='disabled')
        self.pose_progress_var.set('実行中…')

        # 有効カメラのリストを収集
        enabled_panels = [
            (i, self.panels[i])
            for i in range(N_CAMS)
            if self.pose_cam_enabled[i].get() and self.panels[i].caps
        ]

        if not enabled_panels:
            messagebox.showerror('Error', '動画が読み込まれているアクティブカメラがありません。')
            self._pose_running = False
            self.run_mediapipe_btn.config(state='normal')
            return

        self._pose_log_queue = queue.Queue()

        def _worker():
            total = len(enabled_panels)
            for step_i, (cam_i, panel) in enumerate(enabled_panels):
                cam_name = f'cam{cam_i + 1}'
                self._pose_log_queue.put(f'--- {cam_name} ({step_i + 1}/{total}) ---\n')

                # video_segments の構築
                start_s = self.pose_start_frame.get()
                end_s   = self.pose_end_frame.get()
                segments = []
                for seg_i, path in enumerate(panel.paths):
                    seg_start_abs = panel.cum_frames[seg_i]
                    seg_end_abs   = panel.cum_frames[seg_i + 1]
                    cam_start = self.sync_offsets[cam_i] + start_s
                    cam_end   = self.sync_offsets[cam_i] + end_s
                    # このセグメントが範囲と重なるか
                    local_start = max(0, cam_start - seg_start_abs)
                    local_end   = min(panel.segment_frames[seg_i],
                                      cam_end - seg_start_abs)
                    if local_end <= local_start:
                        continue
                    segments.append({
                        'path':  path,
                        'start': local_start,
                        'end':   local_end,
                    })

                if not segments:
                    self._pose_log_queue.put(f'  {cam_name}: 範囲内のセグメントなし、スキップ\n')
                    continue

                config = {
                    'cam_name':       cam_name,
                    'video_segments': segments,
                    'synced_start':   start_s,
                    'det_conf':       self.pose_det_conf.get(),
                    'pres_conf':      self.pose_det_conf.get(),
                    'track_conf':     self.pose_det_conf.get(),
                    # vis_thresh なし: 全ランドマークを保存
                    # 結果は sync_config.json の pose3d.landmarks に保存（CSV不要）
                }

                cfg_file = Path(f'_{cam_name}_mediapipe_config.json')
                import json as _json
                cfg_file.write_text(
                    _json.dumps(config, indent=2, ensure_ascii=False),
                    encoding='utf-8',
                )

                try:
                    proc = subprocess.Popen(
                        [sys.executable, 'recog_mediapipe.py', '--config', str(cfg_file)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, encoding='utf-8', errors='replace',
                    )
                    for line in proc.stdout:
                        self._pose_log_queue.put(strip_ansi(line))
                    proc.wait()
                    cfg_file.unlink(missing_ok=True)
                    if proc.returncode != 0:
                        self._pose_log_queue.put(
                            f'  {cam_name}: 失敗 (code={proc.returncode})\n')
                except Exception as ex:
                    cfg_file.unlink(missing_ok=True)
                    self._pose_log_queue.put(f'  ERROR: {ex}\n')

                self._pose_log_queue.put(f'  {cam_name}: 完了\n')

            self._pose_log_queue.put(('__done__', 'MediaPipe', 0))

        threading.Thread(target=_worker, daemon=True).start()
        self.root.after(100, self._poll_pose_log)

    def _rerun_cam(self):
        """現在プレビュー中のカメラだけ MediaPipe を再実行する。"""
        if self._pose_running:
            messagebox.showinfo('Info', 'すでに実行中です。')
            return

        cam_name = self.pose_preview_cam_var.get()
        cam_i    = int(cam_name.replace('cam', '')) - 1
        panel    = self.panels[cam_i]

        if not panel.caps:
            messagebox.showerror('Error', f'{cam_name} に動画が読み込まれていません。')
            return

        start_s = self.pose_start_frame.get()
        end_s   = self.pose_end_frame.get()
        segments = []
        for seg_i, path in enumerate(panel.paths):
            seg_start_abs = panel.cum_frames[seg_i]
            cam_start = self.sync_offsets[cam_i] + start_s
            cam_end   = self.sync_offsets[cam_i] + end_s
            local_start = max(0, cam_start - seg_start_abs)
            local_end   = min(panel.segment_frames[seg_i], cam_end - seg_start_abs)
            if local_end <= local_start:
                continue
            segments.append({'path': path, 'start': local_start, 'end': local_end})

        if not segments:
            messagebox.showerror('Error', f'{cam_name}: 範囲内のセグメントがありません。')
            return

        config = {
            'cam_name':       cam_name,
            'video_segments': segments,
            'synced_start':   start_s,
            'det_conf':       self.pose_det_conf.get(),
            'pres_conf':      self.pose_det_conf.get(),
            'track_conf':     self.pose_det_conf.get(),
        }

        cfg_file = Path(f'_{cam_name}_rerun_config.json')
        import json as _json
        cfg_file.write_text(
            _json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')

        self._pose_running = True
        self._pose_log_queue = queue.Queue()
        self.pose_progress_var.set(f'{cam_name} 再実行中…')

        def _worker():
            try:
                proc = subprocess.Popen(
                    [sys.executable, 'recog_mediapipe.py', '--config', str(cfg_file)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, encoding='utf-8', errors='replace',
                )
                for line in proc.stdout:
                    self._pose_log_queue.put(strip_ansi(line))
                proc.wait()
                cfg_file.unlink(missing_ok=True)
                self._pose_log_queue.put(('__done__', f'{cam_name} re-run', proc.returncode))
            except Exception as ex:
                cfg_file.unlink(missing_ok=True)
                self._pose_log_queue.put(f'ERROR: {ex}\n')
                self._pose_log_queue.put(('__done__', f'{cam_name} re-run', -1))

        threading.Thread(target=_worker, daemon=True).start()
        self.root.after(100, self._poll_pose_log)

    # ── ポーズプレビューモード ──────────────────────

    def _on_tab_change(self, event=None):
        """ノートブックタブ切替時に各モード（Stick Check / Wand）を制御する。"""
        try:
            idx = self.nb.index(self.nb.select())
        except Exception:
            return

        entering_wand  = (idx == 1)   # Step 2: Wand Annotation
        entering_stick = (idx == 5)   # Step 6: Stick Check

        # ワンドモード終了
        if not entering_wand and self._wand_active:
            self._exit_wand_mode()
            if not entering_stick:
                self._enter_multi_view()

        # Stick Check モード制御
        if entering_stick and self.synced:
            self._enter_pose_preview_mode()
        elif not entering_stick and self.in_pose_preview_mode:
            self._exit_pose_preview_mode(restore_view=False)
            if not entering_wand:
                self._enter_single_view(self._get_cam_idx())

        # ワンドアノテーションモード開始
        if entering_wand:
            self._wand_on_tab_enter()

    def _enter_pose_preview_mode(self):
        """panel_area を [映像 | スティックフィギュア] 横並びに切り替える。"""
        cam_name = self.pose_preview_cam_var.get()
        try:
            cam_idx = int(cam_name.replace('cam', '')) - 1
        except ValueError:
            cam_idx = 0

        self.in_pose_preview_mode = True
        self.single_view_active = True
        self.cam_select_var.set(cam_name)
        self.view_btn.config(text='⊞ All cams')

        self.panel_area.pack_configure(fill='both', expand=True)
        self.nb.pack_configure(fill='x', expand=False)

        # 全パネルを非表示にし選択カメラを左列に
        for p in self.panels:
            p.frame.grid_remove()
        self._skel_lf.grid_remove()
        self._skel_smooth_lf.grid_remove()

        self.panels[cam_idx].frame.grid(
            row=0, column=0, padx=(4, 2), pady=4, sticky='nsew')
        self._skel_lf.grid(
            row=0, column=1, padx=2, pady=4, sticky='nsew')
        self._skel_smooth_lf.grid(
            row=0, column=2, padx=(2, 4), pady=4, sticky='nsew')

        self.panel_area.grid_rowconfigure(0, weight=1)
        for c in range(3):
            self.panel_area.grid_columnconfigure(c, weight=1, minsize=PANEL_W)

        # 高さはここで1回だけ確定させる（Configure ループによる膨張を防ぐ）
        self.root.update_idletasks()
        self._preview_canvas_height = max(PANEL_H, self.panel_area.winfo_height() - 92)

        # 幅方向は Configure バインドで追従（高さは固定値を使う）
        self.panel_area.bind('<Configure>', self._on_panel_area_configure)
        self.root.after(30, lambda: self._resize_pose_preview(cam_idx))
        self._update_skel_canvas(self.sync_pos)
        self._update_smooth_skel_canvas(self.sync_pos)

    def _on_panel_area_configure(self, event=None):
        """panel_area 幅変化時にキャンバス幅を等分配分する（高さは変更しない）。"""
        if not self.in_pose_preview_mode:
            return
        if self._resize_after_id:
            self.root.after_cancel(self._resize_after_id)
        cam_idx = self._get_cam_idx()
        self._resize_after_id = self.root.after(
            80, lambda: self._resize_pose_preview(cam_idx))

    def _exit_pose_preview_mode(self, restore_view: bool = True):
        """ポーズプレビューモードを解除する。"""
        self.in_pose_preview_mode = False
        self.panel_area.unbind('<Configure>')
        if self._resize_after_id:
            self.root.after_cancel(self._resize_after_id)
            self._resize_after_id = None
        self._skel_lf.grid_remove()
        self._skel_smooth_lf.grid_remove()
        if restore_view:
            self._enter_single_view(self._get_cam_idx())

    def _resize_pose_preview(self, cam_idx: int):
        """ポーズプレビューモード時の3キャンバスを等幅に調整する（高さは固定）。"""
        self._resize_after_id = None
        total_w = self.panel_area.winfo_width()
        if total_w <= 1:
            return  # まだレイアウト未確定
        third_w = max(PANEL_W, (total_w - 32) // 3)
        # 高さは入室時に1回だけ確定した値を使用（Configure ループ対策）
        ah = self._preview_canvas_height
        p = self.panels[cam_idx]
        p.canvas.config(width=third_w, height=ah)
        self._skel_canvas.config(width=third_w, height=ah)
        self._skel_smooth_canvas.config(width=third_w, height=ah)
        if p.caps:
            p.show_frame(p.current_frame)
        else:
            p._show_placeholder()

    def _update_skel_canvas(self, sync_pos: int):
        """スティックフィギュアキャンバスをバックグラウンドで更新する。"""
        cam_name = self.pose_preview_cam_var.get()
        try:
            cam_i = int(cam_name.replace('cam', '')) - 1
        except ValueError:
            return
        panel = self.panels[cam_i]
        if not panel.caps:
            return

        abs_frame = self.sync_offsets[cam_i] + sync_pos

        def _load():
            # 動画アスペクト比のみ取得（フレーム読み込み不要）
            with panel._cap_lock:
                if not panel.caps:
                    return
                vid_w = int(panel.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                vid_h = int(panel.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

            cw = int(self._skel_canvas.cget('width'))
            ch = int(self._skel_canvas.cget('height'))
            scale = min(cw / vid_w, ch / vid_h)
            nw = max(1, int(vid_w * scale))
            nh = max(1, int(vid_h * scale))

            vis_thresh = self.stick_thresh_vars.get(cam_name, tk.DoubleVar(value=0.0)).get()
            sub = np.full((nh, nw, 3), 255, dtype=np.uint8)
            sub = self._draw_skeleton(sub, cam_name, sync_pos, nw, nh, vis_thresh)
            canvas_img = np.full((ch, cw, 3), 255, dtype=np.uint8)
            ox, oy = (cw - nw) // 2, (ch - nh) // 2
            canvas_img[oy:oy + nh, ox:ox + nw] = sub

            pil_skel = Image.fromarray(canvas_img)
            self.root.after(0, lambda: self._display_skel_canvas(pil_skel, cw, ch))

        threading.Thread(target=_load, daemon=True).start()

    def _display_skel_canvas(self, pil_img: Image.Image, cw: int, ch: int):
        """スティックフィギュアキャンバスをメインスレッドで更新する。"""
        self._pose_skel_photo = ImageTk.PhotoImage(pil_img)
        self._skel_canvas.delete('all')
        self._skel_canvas.create_image(
            cw // 2, ch // 2, anchor='center', image=self._pose_skel_photo)

    # ── 補間・平滑化スティックフィギュア ──────────────────────────────

    def _on_smooth_param_change(self, _=None):
        """平滑化パラメータ変更時: キャッシュ無効化 + 表示値更新 + 再描画。"""
        self._smooth_gap_disp.set(str(int(self.smooth_max_gap_var.get())))
        self._smooth_win_disp.set(str(int(self.smooth_window_var.get())))
        self._pose_smooth_cache.clear()
        if self.in_pose_preview_mode:
            self._update_smooth_skel_canvas(self.sync_pos)

    def _get_smoothed_df(self, cam_name: str):
        """指定カメラの補間・平滑化済み DataFrame を返す（キャッシュあり）。"""
        import pandas as pd
        max_gap = int(self.smooth_max_gap_var.get())
        window  = max(1, int(self.smooth_window_var.get()))
        key = (cam_name, max_gap, window)
        if key in self._pose_smooth_cache:
            return self._pose_smooth_cache[key]

        # 生データを取得（未ロードなら読み込む）
        if cam_name not in self._pose_csv_data:
            try:
                cfg = json.loads(Path(self._json_path).read_text(encoding='utf-8'))
                lm_entry = cfg.get('pose3d', {}).get('landmarks', {}).get(cam_name)
                if not lm_entry:
                    return None
                df_raw = pd.DataFrame(lm_entry['data'], columns=lm_entry['columns'])
                self._pose_csv_data[cam_name] = df_raw
            except Exception:
                return None

        df = self._pose_csv_data[cam_name].copy()
        coord_cols = [c for c in df.columns
                      if c.endswith('_x') or c.endswith('_y') or c.endswith('_z')]

        # 線形補間（最大 max_gap フレームの連続欠損のみ）
        if max_gap > 0:
            df[coord_cols] = df[coord_cols].interpolate(
                method='linear', limit=max_gap, limit_direction='both')

        # 移動平均平滑化
        if window > 1:
            df[coord_cols] = (
                df[coord_cols]
                .rolling(window=window, center=True, min_periods=1)
                .mean()
            )

        self._pose_smooth_cache[key] = df
        return df

    def _update_smooth_skel_canvas(self, sync_pos: int):
        """平滑化スティックフィギュアキャンバスを更新する。"""
        cam_name = self.pose_preview_cam_var.get()
        try:
            cam_i = int(cam_name.replace('cam', '')) - 1
        except ValueError:
            return
        panel = self.panels[cam_i]
        if not panel.caps:
            return

        def _load():
            with panel._cap_lock:
                if not panel.caps:
                    return
                vid_w = int(panel.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                vid_h = int(panel.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

            cw = int(self._skel_smooth_canvas.cget('width'))
            ch = int(self._skel_smooth_canvas.cget('height'))
            scale = min(cw / vid_w, ch / vid_h)
            nw = max(1, int(vid_w * scale))
            nh = max(1, int(vid_h * scale))

            vis_thresh = self.stick_thresh_vars.get(cam_name, tk.DoubleVar(value=0.0)).get()
            df_smooth = self._get_smoothed_df(cam_name)
            sub = np.full((nh, nw, 3), 240, dtype=np.uint8)  # 淡い背景
            if df_smooth is not None:
                sub = self._draw_skeleton_from_df(
                    sub, df_smooth, sync_pos, nw, nh, vis_thresh, color_joints=(0, 130, 0))
            canvas_img = np.full((ch, cw, 3), 240, dtype=np.uint8)
            ox, oy = (cw - nw) // 2, (ch - nh) // 2
            canvas_img[oy:oy + nh, ox:ox + nw] = sub

            pil_img = Image.fromarray(canvas_img)
            self.root.after(0, lambda: self._display_smooth_canvas(pil_img, cw, ch))

        threading.Thread(target=_load, daemon=True).start()

    def _display_smooth_canvas(self, pil_img: Image.Image, cw: int, ch: int):
        self._pose_skel_smooth_photo = ImageTk.PhotoImage(pil_img)
        self._skel_smooth_canvas.delete('all')
        self._skel_smooth_canvas.create_image(
            cw // 2, ch // 2, anchor='center', image=self._pose_skel_smooth_photo)

    def _draw_skeleton_from_df(self, img_rgb: np.ndarray, df, synced_frame: int,
                               img_w: int, img_h: int, vis_thresh: float = 0.0,
                               color_joints=(0, 0, 200)) -> np.ndarray:
        """DataFrame の特定フレーム行からスケルトンを描画して返す。"""
        import math
        rows = df[df['frame'] == synced_frame]
        if len(rows) == 0:
            return img_rgb
        row = rows.iloc[0]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        def get_pt(lm_name):
            x_col, y_col, v_col = f'{lm_name}_x', f'{lm_name}_y', f'{lm_name}_v'
            if x_col not in row or y_col not in row:
                return None
            x, y = row[x_col], row[y_col]
            if math.isnan(x) or math.isnan(y):
                return None
            if v_col in row:
                v = row[v_col]
                if not math.isnan(v) and v < vis_thresh:
                    return None
            return (int(x * img_w), int(y * img_h))

        bone_color = tuple(int(c * 0.6) for c in color_joints[::-1])
        for a, b in self.POSE_2D_CONNECTIONS:
            pa, pb = get_pt(a), get_pt(b)
            if pa is not None and pb is not None:
                cv2.line(img_bgr, pa, pb, bone_color, 3)

        for col in df.columns:
            if col.endswith('_x'):
                pt = get_pt(col[:-2])
                if pt is not None:
                    cv2.circle(img_bgr, pt, 5, color_joints[::-1], -1)

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    POSE_2D_CONNECTIONS = [
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

    def _draw_skeleton(self, img_rgb: np.ndarray, cam_name: str,
                       synced_frame: int, img_w: int, img_h: int,
                       vis_thresh: float = 0.0) -> np.ndarray:
        """CSV データからスケルトンを img に描画して返す。"""
        if cam_name not in self._pose_csv_data:
            # sync_config.json の pose3d.landmarks から読み込む
            try:
                import pandas as pd
                cfg = json.loads(Path(self._json_path).read_text(encoding='utf-8'))
                lm_entry = cfg.get('pose3d', {}).get('landmarks', {}).get(cam_name)
                if lm_entry:
                    df = pd.DataFrame(lm_entry['data'], columns=lm_entry['columns'])
                    self._pose_csv_data[cam_name] = df
                else:
                    return img_rgb
            except Exception:
                return img_rgb

        df = self._pose_csv_data[cam_name]
        row = df[df['frame'] == synced_frame]
        if len(row) == 0:
            return img_rgb

        row = row.iloc[0]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        import math

        def get_pt(lm_name):
            x_col, y_col, v_col = f'{lm_name}_x', f'{lm_name}_y', f'{lm_name}_v'
            if x_col not in row or y_col not in row:
                return None
            x, y = row[x_col], row[y_col]
            if math.isnan(x) or math.isnan(y):
                return None
            # visibility フィルタ
            if v_col in row:
                v = row[v_col]
                if not math.isnan(v) and v < vis_thresh:
                    return None
            return (int(x * img_w), int(y * img_h))

        for a, b in self.POSE_2D_CONNECTIONS:
            pa, pb = get_pt(a), get_pt(b)
            if pa is not None and pb is not None:
                cv2.line(img_bgr, pa, pb, (200, 80, 0), 3)

        for col in df.columns:
            if col.endswith('_x'):
                lm = col[:-2]
                pt = get_pt(lm)
                if pt is not None:
                    cv2.circle(img_bgr, pt, 5, (0, 0, 200), -1)

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _on_pose_step_done(self, step: str, returncode: int):
        """MediaPipe 実行完了コールバック（Step 4 ログ用）。"""
        if returncode == 0:
            self._pose_log_write(f'\n[{step}] 完了\n')
            self.pose_progress_var.set('完了')
        else:
            self._pose_log_write(f'\n[{step}] 失敗 (code={returncode})\n')
            self.pose_progress_var.set(f'失敗 (code={returncode})')
        self._pose_running = False
        self.run_mediapipe_btn.config(state='normal')

    # ── Step 5: Stick Check ─────────────────────────

    def _build_stick_check_tab(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text='  Step 6: Stick Check  ')

        ttk.Label(tab,
            text='▲ 上パネルに映像 + スティックフィギュアをフレーム同期表示します（スライダーで操作）',
            foreground='#0088cc').pack(anchor='w', pady=(0, 6))

        # カメラ選択 / 再実行 / 保存 を1行に
        ctrl_row = ttk.Frame(tab)
        ctrl_row.pack(fill='x', pady=(0, 8))
        ttk.Label(ctrl_row, text='表示カメラ:').pack(side='left')
        cam_combo = ttk.Combobox(
            ctrl_row, textvariable=self.pose_preview_cam_var,
            values=[f'cam{i + 1}' for i in range(N_CAMS)],
            width=6, state='readonly',
        )
        cam_combo.pack(side='left', padx=(4, 8))
        cam_combo.bind(
            '<<ComboboxSelected>>',
            lambda e: self._enter_pose_preview_mode() if self.in_pose_preview_mode else None,
        )
        ttk.Button(ctrl_row, text='Re-run cam', command=self._rerun_cam).pack(side='left', padx=(0, 12))
        ttk.Separator(ctrl_row, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(ctrl_row, text='閾値を保存', command=self._save_stick_thresholds).pack(side='left', padx=(0, 8))
        self._stick_thresh_status = tk.StringVar(value='')
        ttk.Label(ctrl_row, textvariable=self._stick_thresh_status,
                  foreground='#0088cc').pack(side='left')

        # カメラ別 visibility 閾値 (2列グリッド)
        thresh_lf = ttk.LabelFrame(tab, text='カメラ別 Visibility 閾値', padding=6)
        thresh_lf.pack(fill='x')

        # 表示用 StringVar（小数2桁）
        self._thresh_disp_vars = {}
        for i in range(N_CAMS):
            cam = f'cam{i + 1}'
            disp = tk.StringVar(value='0.50')
            self._thresh_disp_vars[cam] = disp
            self.stick_thresh_vars[cam].trace_add(
                'write',
                lambda *_, c=cam: self._thresh_disp_vars[c].set(
                    f'{self.stick_thresh_vars[c].get():.2f}'),
            )

        cols = 3
        for i in range(N_CAMS):
            cam = f'cam{i + 1}'
            col = (i % cols) * 3
            row_idx = i // cols
            thresh_lf.grid_columnconfigure(col + 1, weight=1)

            ttk.Label(thresh_lf, text=f'{cam}:', width=5, anchor='e').grid(
                row=row_idx, column=col, padx=(8 if col else 0, 2), pady=3, sticky='e')
            ttk.Scale(
                thresh_lf, from_=0.0, to=1.0, orient='horizontal',
                variable=self.stick_thresh_vars[cam],
                command=lambda val, c=cam: self._on_thresh_change(c),
            ).grid(row=row_idx, column=col + 1, sticky='ew', padx=2, pady=3)
            ttk.Label(thresh_lf, textvariable=self._thresh_disp_vars[cam],
                      width=4, anchor='w').grid(
                row=row_idx, column=col + 2, padx=(2, 12), pady=3, sticky='w')

        # 補間・平滑化パラメータスライダー
        smooth_lf = ttk.LabelFrame(tab, text='補間・平滑化パラメータ（右パネル）', padding=6)
        smooth_lf.pack(fill='x', pady=(6, 0))
        smooth_lf.columnconfigure(1, weight=1)

        self._smooth_gap_disp = tk.StringVar(value=str(self.smooth_max_gap_var.get()))
        self._smooth_win_disp = tk.StringVar(value=str(self.smooth_window_var.get()))

        ttk.Label(smooth_lf, text='補間 最大欠損フレーム数:', anchor='e').grid(
            row=0, column=0, sticky='e', padx=(4, 4), pady=3)
        ttk.Scale(
            smooth_lf, from_=0, to=30, orient='horizontal',
            variable=self.smooth_max_gap_var,
            command=self._on_smooth_param_change,
        ).grid(row=0, column=1, sticky='ew', padx=4, pady=3)
        ttk.Label(smooth_lf, textvariable=self._smooth_gap_disp,
                  width=4, anchor='w').grid(row=0, column=2, padx=(2, 8), pady=3)

        ttk.Label(smooth_lf, text='平滑化 ウィンドウ幅 (フレーム):', anchor='e').grid(
            row=1, column=0, sticky='e', padx=(4, 4), pady=3)
        ttk.Scale(
            smooth_lf, from_=1, to=31, orient='horizontal',
            variable=self.smooth_window_var,
            command=self._on_smooth_param_change,
        ).grid(row=1, column=1, sticky='ew', padx=4, pady=3)
        ttk.Label(smooth_lf, textvariable=self._smooth_win_disp,
                  width=4, anchor='w').grid(row=1, column=2, padx=(2, 8), pady=3)

    # ── Wand Annotation タブ ─────────────────────────

    def _build_wand_tab(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text='  Step 2: Wand Annotation  ')

        # ── カメラ・ポーズ選択 ─────────────────────────
        top = ttk.Frame(tab)
        top.pack(fill='x', pady=(0, 4))

        ttk.Label(top, text='Camera:').pack(side='left')
        wand_cam_combo = ttk.Combobox(
            top, textvariable=self.wand_cam_var,
            values=[f'cam{i + 1}' for i in range(N_CAMS)],
            width=6, state='readonly',
        )
        wand_cam_combo.pack(side='left', padx=(4, 8))
        wand_cam_combo.bind('<<ComboboxSelected>>', self._wand_on_cam_change)

        ttk.Label(top, text='Pose:').pack(side='left')
        ttk.Button(top, text='◀', width=3, command=self._wand_prev_pose).pack(side='left', padx=2)
        self._wand_pose_label = ttk.Label(top, text='pose1  (1/5)', width=16)
        self._wand_pose_label.pack(side='left')
        ttk.Button(top, text='▶ 次ポーズ', command=self._wand_next_pose).pack(side='left', padx=2)

        ttk.Separator(top, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(top, text='📂 CSV 読込', command=self._wand_load_csv).pack(side='left', padx=2)
        ttk.Button(top, text='💾 CSV 保存', command=self._wand_save_csv).pack(side='left', padx=2)
        self._wand_save_status = tk.StringVar(value='')
        ttk.Label(top, textvariable=self._wand_save_status,
                  foreground='#0088cc').pack(side='left', padx=4)

        # ── フレームナビゲーション ─────────────────────
        nav = ttk.Frame(tab)
        nav.pack(fill='x', pady=(0, 4))

        ttk.Button(nav, text='◀◀', width=3,
                   command=lambda: self._wand_seek(-100)).pack(side='left')
        ttk.Button(nav, text='◀',  width=3,
                   command=lambda: self._wand_seek(-10)).pack(side='left', padx=2)
        ttk.Button(nav, text='◀',  width=3,
                   command=lambda: self._wand_seek(-1)).pack(side='left', padx=2)
        self._wand_frame_var = tk.IntVar(value=0)
        self._wand_frame_entry = ttk.Entry(
            nav, textvariable=self._wand_frame_var, width=8, justify='center')
        self._wand_frame_entry.pack(side='left', padx=2)
        self._wand_frame_entry.bind(
            '<Return>', lambda e: self._wand_show_frame(self._wand_frame_var.get()))
        self._wand_total_label = ttk.Label(nav, text='/ --', width=8)
        self._wand_total_label.pack(side='left')
        ttk.Button(nav, text='▶',  width=3,
                   command=lambda: self._wand_seek(1)).pack(side='left', padx=2)
        ttk.Button(nav, text='▶',  width=3,
                   command=lambda: self._wand_seek(10)).pack(side='left', padx=2)
        ttk.Button(nav, text='▶▶', width=3,
                   command=lambda: self._wand_seek(100)).pack(side='left')

        ttk.Separator(nav, orient='vertical').pack(side='left', fill='y', padx=8)
        self._wand_click_btn = ttk.Button(
            nav, text='Click Mode: OFF', command=self._wand_toggle_click)
        self._wand_click_btn.pack(side='left', padx=4)
        ttk.Button(nav, text='Reset', command=self._wand_reset).pack(side='left', padx=2)

        ttk.Separator(nav, orient='vertical').pack(side='left', fill='y', padx=8)
        self._wand_pts_label = ttk.Label(
            nav, text='0.0m✗  0.5m✗  1.0m✗  1.5m✗', width=30)
        self._wand_pts_label.pack(side='left')

        # ── 進捗グリッド ───────────────────────────────
        prog_lf = ttk.LabelFrame(tab, text='Annotation Progress', padding=4)
        prog_lf.pack(fill='x', pady=(4, 0))
        self._wand_prog_labels = {}
        for i in range(N_CAMS):
            cam = f'cam{i + 1}'
            lbl = ttk.Label(prog_lf,
                            text=f'{cam}: 0/{len(self.wand_pose_names)}', width=12)
            lbl.grid(row=0, column=i, padx=6, pady=2)
            self._wand_prog_labels[cam] = lbl

        ttk.Label(
            tab,
            text='▲ 上パネルにワンド映像を表示します。Click Mode ON → 4点クリック → 自動で次ポーズへ',
            foreground='gray',
        ).pack(anchor='w', pady=(6, 0))

    # ── Wand モード制御 ────────────────────────────────

    def _wand_on_tab_enter(self):
        if not self.synced:
            return
        if not self._wand_active:
            self._enter_wand_mode()

    def _enter_wand_mode(self):
        self._wand_active    = True
        self.single_view_active = True
        self.panel_area.pack_configure(fill='both', expand=True)
        self.nb.pack_configure(fill='x', expand=False)

        for p in self.panels:
            p.frame.grid_remove()
        self._skel_lf.grid_remove()

        self._wand_panel_lf.grid(
            row=0, column=0, columnspan=COLS, padx=4, pady=4, sticky='nsew')
        self.panel_area.grid_rowconfigure(0, weight=1)
        for c in range(COLS):
            self.panel_area.grid_columnconfigure(c, weight=1)

        self.root.update_idletasks()
        self._wand_load_camera(self.wand_cam_var.get())

    def _exit_wand_mode(self):
        if not self._wand_active:
            return
        self._wand_active       = False
        self.single_view_active = False
        self._wand_panel_lf.grid_remove()
        if self._wand_cap is not None:
            self._wand_cap.release()
            self._wand_cap      = None
            self._wand_cap_path = ''

    # ── カメラ読み込み・フレーム表示 ──────────────────

    def _wand_load_camera(self, cam_name: str):
        cam_idx = int(cam_name.replace('cam', '')) - 1
        panel   = self.panels[cam_idx]
        if not panel.paths:
            self._wand_show_placeholder(f'{cam_name}: 動画未読込')
            return
        path = panel.paths[0]
        if self._wand_cap is not None and self._wand_cap_path == path:
            self._wand_show_frame(self._wand_frame_idx)
            return
        if self._wand_cap is not None:
            self._wand_cap.release()
        self._wand_cap       = cv2.VideoCapture(path)
        self._wand_cap_path  = path
        self._wand_total_frames = int(self._wand_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._wand_total_label.config(text=f'/ {self._wand_total_frames - 1}')
        # sync offset 位置からスタート
        start = max(0, min(self.sync_offsets[cam_idx], self._wand_total_frames - 1))
        self._wand_show_frame(start)
        self._wand_update_pose_label()
        self._wand_update_progress()

    def _wand_show_frame(self, frame_idx: int):
        if self._wand_cap is None:
            return
        frame_idx = max(0, min(frame_idx, self._wand_total_frames - 1))
        self._wand_frame_idx = frame_idx
        self._wand_frame_var.set(frame_idx)

        self._wand_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = self._wand_cap.read()
        if not ret:
            return

        cam_name  = self.wand_cam_var.get()
        pose_name = self.wand_pose_names[self.wand_pose_idx]
        img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_rgb.shape[:2]

        # 保存済み点（緑）を描画
        for pt_i, label in enumerate(self.wand_point_labels):
            key = (cam_name, pose_name, label)
            if key in self.wand_annotations and pt_i >= len(self.wand_clicked_pts):
                u, v = self.wand_annotations[key]['u'], self.wand_annotations[key]['v']
                cv2.circle(img_rgb, (u, v), 8, (50, 210, 50), -1)
                cv2.putText(img_rgb, label, (u + 10, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 210, 50), 2)

        # クリック中の点（赤橙）を描画
        for pt_i, (u, v) in enumerate(self.wand_clicked_pts):
            label = self.wand_point_labels[pt_i]
            cv2.circle(img_rgb, (u, v), 8, (255, 90, 0), -1)
            cv2.putText(img_rgb, label, (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 90, 0), 2)

        # ステータスオーバーレイ
        if self.wand_click_mode:
            n = len(self.wand_clicked_pts)
            if n < 4:
                txt   = f'[CLICK] Next: {self.wand_point_labels[n]}  ({n + 1}/4)'
                color = (255, 165, 0)
            else:
                txt   = 'All 4 done!'
                color = (50, 220, 50)
        else:
            txt   = 'Click Mode OFF  |  Press [Click Mode] to start'
            color = (160, 160, 160)
        cv2.putText(img_rgb, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(img_rgb, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        info = f'{cam_name}  {pose_name}  frame:{frame_idx}'
        cv2.putText(img_rgb, info, (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(img_rgb, info, (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

        # キャンバスにフィット
        cw = self._wand_panel_canvas.winfo_width()  or 1280
        ch = self._wand_panel_canvas.winfo_height() or 720
        scale = min(cw / w_img, ch / h_img)
        nw    = max(1, int(w_img * scale))
        nh    = max(1, int(h_img * scale))
        ox    = (cw - nw) // 2
        oy    = (ch - nh) // 2

        self._wand_img_scale  = scale
        self._wand_img_offset = (ox, oy)
        self._wand_img_size   = (w_img, h_img)

        pil_img = Image.fromarray(cv2.resize(img_rgb, (nw, nh)))
        self._wand_photo = ImageTk.PhotoImage(pil_img)
        self._wand_panel_canvas.delete('all')
        self._wand_panel_canvas.create_image(ox, oy, anchor='nw', image=self._wand_photo)

        self._wand_update_pts_label()

    def _wand_show_placeholder(self, text: str = 'No video'):
        cw = self._wand_panel_canvas.winfo_width()  or 800
        ch = self._wand_panel_canvas.winfo_height() or 450
        self._wand_panel_canvas.delete('all')
        self._wand_panel_canvas.create_text(
            cw // 2, ch // 2, text=text,
            fill='#555', font=('Arial', 14), justify='center')

    # ── クリック・操作 ─────────────────────────────────

    def _wand_on_canvas_click(self, event):
        if not self.wand_click_mode or len(self.wand_clicked_pts) >= 4:
            return
        ox, oy    = self._wand_img_offset
        scale     = self._wand_img_scale
        w_img, h_img = self._wand_img_size
        img_x = int((event.x - ox) / scale)
        img_y = int((event.y - oy) / scale)
        if not (0 <= img_x < w_img and 0 <= img_y < h_img):
            return
        self.wand_clicked_pts.append((img_x, img_y))
        self._wand_show_frame(self._wand_frame_idx)
        if len(self.wand_clicked_pts) == 4:
            self._wand_confirm_pose()

    def _wand_confirm_pose(self):
        """現在の 4 点クリックをアノテーションに保存して次ポーズへ。"""
        if len(self.wand_clicked_pts) != 4:
            messagebox.showwarning('Warning', '4 点すべてクリックしてください。')
            return
        cam_name  = self.wand_cam_var.get()
        pose_name = self.wand_pose_names[self.wand_pose_idx]
        for i, label in enumerate(self.wand_point_labels):
            u, v = self.wand_clicked_pts[i]
            self.wand_annotations[(cam_name, pose_name, label)] = {
                'frame': self._wand_frame_idx,
                'u': u,
                'v': v,
            }
        self.wand_click_mode = False
        self._wand_click_btn.config(text='Click Mode: OFF')
        self.wand_clicked_pts = []
        self._wand_update_progress()
        if self.wand_pose_idx < len(self.wand_pose_names) - 1:
            self.wand_pose_idx += 1
            self._wand_update_pose_label()
        else:
            messagebox.showinfo('完了', f'{cam_name} の全ポーズ（{len(self.wand_pose_names)}）完了！')
        self._wand_show_frame(self._wand_frame_idx)

    def _wand_seek(self, delta: int):
        self._wand_show_frame(self._wand_frame_idx + delta)

    def _wand_toggle_click(self):
        self.wand_click_mode = not self.wand_click_mode
        if self.wand_click_mode:
            self.wand_clicked_pts = []
            self._wand_click_btn.config(text='Click Mode: ON')
        else:
            self._wand_click_btn.config(text='Click Mode: OFF')
        self._wand_show_frame(self._wand_frame_idx)

    def _wand_reset(self):
        """現在の (cam, pose) のクリック & アノテーションをリセット。"""
        self.wand_clicked_pts = []
        self.wand_click_mode  = False
        self._wand_click_btn.config(text='Click Mode: OFF')
        cam  = self.wand_cam_var.get()
        pose = self.wand_pose_names[self.wand_pose_idx]
        for label in self.wand_point_labels:
            self.wand_annotations.pop((cam, pose, label), None)
        self._wand_update_progress()
        self._wand_show_frame(self._wand_frame_idx)

    def _wand_next_pose(self):
        if len(self.wand_clicked_pts) == 4:
            self._wand_confirm_pose()
            return
        if self.wand_pose_idx < len(self.wand_pose_names) - 1:
            self.wand_pose_idx   += 1
            self.wand_clicked_pts = []
            self.wand_click_mode  = False
            self._wand_click_btn.config(text='Click Mode: OFF')
            self._wand_update_pose_label()
            self._wand_show_frame(self._wand_frame_idx)

    def _wand_prev_pose(self):
        if self.wand_pose_idx > 0:
            self.wand_pose_idx   -= 1
            self.wand_clicked_pts = []
            self.wand_click_mode  = False
            self._wand_click_btn.config(text='Click Mode: OFF')
            self._wand_update_pose_label()
            self._wand_show_frame(self._wand_frame_idx)

    def _wand_on_cam_change(self, _=None):
        self.wand_pose_idx   = 0
        self.wand_clicked_pts = []
        self.wand_click_mode  = False
        self._wand_click_btn.config(text='Click Mode: OFF')
        self._wand_update_pose_label()
        if self._wand_active:
            self._wand_load_camera(self.wand_cam_var.get())

    # ── ラベル更新 ─────────────────────────────────────

    def _wand_update_pose_label(self):
        pose  = self.wand_pose_names[self.wand_pose_idx]
        total = len(self.wand_pose_names)
        self._wand_pose_label.config(
            text=f'{pose}  ({self.wand_pose_idx + 1}/{total})')

    def _wand_update_pts_label(self):
        cam  = self.wand_cam_var.get()
        pose = self.wand_pose_names[self.wand_pose_idx]
        parts = []
        for i, label in enumerate(self.wand_point_labels):
            if i < len(self.wand_clicked_pts):
                parts.append(f'{label}●')
            elif (cam, pose, label) in self.wand_annotations:
                parts.append(f'{label}✓')
            else:
                parts.append(f'{label}✗')
        self._wand_pts_label.config(text='  '.join(parts))

    def _wand_update_progress(self):
        for i in range(N_CAMS):
            cam  = f'cam{i + 1}'
            done = sum(
                1 for p in self.wand_pose_names
                if all((cam, p, lbl) in self.wand_annotations
                       for lbl in self.wand_point_labels)
            )
            total = len(self.wand_pose_names)
            lbl   = self._wand_prog_labels.get(cam)
            if lbl:
                lbl.config(
                    text=f'{cam}: {done}/{total}',
                    foreground=(
                        '#00aa00' if done == total
                        else 'gray' if done == 0 else 'black'
                    ),
                )

    # ── CSV 保存 / 読込 ────────────────────────────────

    def _wand_save_csv(self):
        if not self.wand_annotations:
            messagebox.showerror('Error', 'アノテーションがありません。')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')],
            initialfile='wand_annotations.csv',
        )
        if not path:
            return
        import pandas as pd
        rows = []
        for (cam, pose, label), ann in sorted(self.wand_annotations.items()):
            rows.append({
                'camera':      cam,
                'pose':        pose,
                'point_label': label,
                'frame':       ann['frame'],
                'u':           ann['u'],
                'v':           ann['v'],
            })
        pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8')
        self.calib_wand_csv.set(path)   # キャリブタブに自動反映
        self._wand_save_status.set(f'保存: {Path(path).name}')
        self.root.after(3000, lambda: self._wand_save_status.set(''))

    def _wand_load_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[('CSV', '*.csv')],
            initialfile='wand_annotations.csv',
        )
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path, encoding='utf-8')
            self.wand_annotations.clear()
            for _, row in df.iterrows():
                key = (str(row['camera']), str(row['pose']), str(row['point_label']))
                self.wand_annotations[key] = {
                    'frame': int(row['frame']),
                    'u':     int(row['u']),
                    'v':     int(row['v']),
                }
            self._wand_update_progress()
            if self._wand_active:
                self._wand_show_frame(self._wand_frame_idx)
            self.calib_wand_csv.set(path)
            self._wand_save_status.set(f'読込: {Path(path).name}')
            self.root.after(3000, lambda: self._wand_save_status.set(''))
        except Exception as ex:
            messagebox.showerror('Error', f'CSV 読み込み失敗: {ex}')

    def _on_thresh_change(self, cam_name: str):
        """スライダーで閾値が変わったときにスティックを再描画する。"""
        if self.in_pose_preview_mode and self.pose_preview_cam_var.get() == cam_name:
            self._update_skel_canvas(self.sync_pos)
            self._update_smooth_skel_canvas(self.sync_pos)

    def _save_stick_thresholds(self):
        """カメラ別閾値を sync_config.json に保存する。"""
        try:
            cfg_p = Path(self._json_path)
            cfg = json.loads(cfg_p.read_text(encoding='utf-8')) if cfg_p.exists() else {}
            cfg.setdefault('pose3d', {})['cam_thresholds'] = {
                cam: var.get() for cam, var in self.stick_thresh_vars.items()
            }
            cfg_p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
            self._stick_thresh_status.set('保存しました')
            self.root.after(3000, lambda: self._stick_thresh_status.set(''))
        except Exception as ex:
            self._stick_thresh_status.set(f'エラー: {ex}')

    # ── Step 6: 3D 再構成 ───────────────────────────

    def _build_pose5_tab(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text='  Step 7: 3D Recon  ')

        desc_lf = ttk.LabelFrame(tab, text='Pipeline', padding=6)
        desc_lf.pack(fill='x', pady=(0, 6))
        ttk.Label(desc_lf, text=(
            'triangulate_only.py  →  process_landmarks_3d.py  →  make_3d_plot.py  →  ブラウザ表示'
        )).pack(anchor='w')

        # 入力データ選択（生データ or 補間+平滑化）
        data_lf = ttk.LabelFrame(tab, text='入力データ選択', padding=6)
        data_lf.pack(fill='x', pady=(0, 6))
        ttk.Radiobutton(
            data_lf, text='生データ（Raw）をそのまま三角測量',
            variable=self.recon_data_mode, value='raw',
        ).pack(side='left', padx=(0, 16))
        ttk.Radiobutton(
            data_lf, text='補間 + 平滑化後のデータで三角測量  （Stick Check のパラメータを使用）',
            variable=self.recon_data_mode, value='smooth',
        ).pack(side='left')

        run_row = ttk.Frame(tab)
        run_row.pack(fill='x', pady=(0, 6))
        self.run_3d_btn = ttk.Button(
            run_row, text='▶  Run 3D Reconstruction',
            command=self.run_3d_reconstruction)
        self.run_3d_btn.pack(side='left', padx=(0, 8))
        self.recon_status_var = tk.StringVar(value='')
        ttk.Label(run_row, textvariable=self.recon_status_var,
                  foreground='#0088cc').pack(side='left')

        log_lf = ttk.LabelFrame(tab, text='Output', padding=4)
        log_lf.pack(fill='both', expand=True)
        self.recon_log = tk.Text(
            log_lf, wrap='none', font=('Courier', 9),
            bg='#1e1e1e', fg='#d4d4d4', state='disabled',
        )
        sb_y = ttk.Scrollbar(log_lf, orient='vertical', command=self.recon_log.yview)
        sb_x = ttk.Scrollbar(log_lf, orient='horizontal', command=self.recon_log.xview)
        self.recon_log.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.pack(side='right', fill='y')
        sb_x.pack(side='bottom', fill='x')
        self.recon_log.pack(fill='both', expand=True)

    def _recon_log_write(self, text: str):
        self.recon_log.config(state='normal')
        self.recon_log.insert('end', text)
        self.recon_log.see('end')
        self.recon_log.config(state='disabled')

    def _poll_recon_log(self):
        if self._recon_log_queue is None:
            return
        try:
            while True:
                item = self._recon_log_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == '__done__':
                    self._on_recon_done(item[1], item[2])
                    return
                self._recon_log_write(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_recon_log)

    def _on_recon_done(self, step: str, returncode: int):
        if returncode == 0:
            self._recon_log_write(f'\n[{step}] 完了\n')
            self.recon_status_var.set('完了')
        else:
            self._recon_log_write(f'\n[{step}] 失敗 (code={returncode})\n')
            self.recon_status_var.set(f'失敗 (code={returncode})')
        self._pose_running = False
        self.run_3d_btn.config(state='normal')

    def run_3d_reconstruction(self):
        if self._pose_running:
            messagebox.showinfo('Info', 'すでに実行中です。')
            return

        self._autosave()
        self._pose_running = True
        self.run_3d_btn.config(state='disabled')
        self.recon_status_var.set('3D 再構成中…')

        self.recon_log.config(state='normal')
        self.recon_log.delete('1.0', 'end')
        self.recon_log.config(state='disabled')

        self._recon_log_queue = queue.Queue()

        def _run_step(cmd: list, step_name: str) -> int:
            self._recon_log_queue.put(f'--- {step_name} ---\n')
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, encoding='utf-8', errors='replace',
                )
                for line in proc.stdout:
                    self._recon_log_queue.put(strip_ansi(line))
                proc.wait()
                return proc.returncode
            except Exception as ex:
                self._recon_log_queue.put(f'ERROR: {ex}\n')
                return -1

        def _worker():
            import json as _json
            import pandas as _pd

            sf = self.pose_start_frame.get()
            ef = self.pose_end_frame.get()
            tri_csv  = f'3d_{sf}_{ef}.csv'
            proc_csv = f'3d_{sf}_{ef}_processed.csv'
            html_out = f'3d_{sf}_{ef}.html'

            use_smooth  = (self.recon_data_mode.get() == 'smooth')
            tmp_cfg_path = Path('_recon_smooth_config.json')

            if use_smooth:
                # 平滑化済み landmarks を一時 sync_config に書き出す
                max_gap = int(self.smooth_max_gap_var.get())
                window  = max(1, int(self.smooth_window_var.get()))
                self._recon_log_queue.put(
                    f'補間 (最大欠損={max_gap}) + 平滑化 (window={window}) を適用中…\n')
                try:
                    base_cfg = _json.loads(
                        Path(self._json_path).read_text(encoding='utf-8'))
                    lm_store = base_cfg.get('pose3d', {}).get('landmarks', {})
                    smooth_lm = {}
                    for cam_name, lm_entry in lm_store.items():
                        df = _pd.DataFrame(lm_entry['data'], columns=lm_entry['columns'])
                        coord_cols = [c for c in df.columns
                                      if c.endswith('_x') or c.endswith('_y')
                                      or c.endswith('_z')]
                        if max_gap > 0:
                            df[coord_cols] = df[coord_cols].interpolate(
                                method='linear', limit=max_gap, limit_direction='both')
                        if window > 1:
                            df[coord_cols] = (
                                df[coord_cols]
                                .rolling(window=window, center=True, min_periods=1)
                                .mean()
                            )
                        smooth_lm[cam_name] = {
                            'columns': df.columns.tolist(),
                            'data':    df.values.tolist(),
                        }
                        self._recon_log_queue.put(f'  {cam_name}: {len(df)} フレーム 平滑化済み\n')
                    tmp_cfg = dict(base_cfg)
                    tmp_cfg['pose3d'] = dict(base_cfg.get('pose3d', {}))
                    tmp_cfg['pose3d']['landmarks'] = smooth_lm
                    tmp_cfg_path.write_text(
                        _json.dumps(tmp_cfg, indent=2, ensure_ascii=False), encoding='utf-8')
                    tri_cmd = [sys.executable, 'triangulate_only.py',
                               '--config', str(tmp_cfg_path)]
                except Exception as ex:
                    self._recon_log_queue.put(f'ERROR (平滑化): {ex}\n')
                    self._recon_log_queue.put(('__done__', '3D reconstruction', -1))
                    return
            else:
                tri_cmd = [sys.executable, 'triangulate_only.py']

            rc = _run_step(tri_cmd, 'triangulate_only')
            tmp_cfg_path.unlink(missing_ok=True)
            if rc != 0:
                self._recon_log_queue.put(('__done__', '3D reconstruction', rc))
                return
            rc = _run_step([
                sys.executable, 'process_landmarks_3d.py',
                '--input', tri_csv,
                '--output', proc_csv,
                '--drop-first-frames', '0',
            ], 'process_landmarks_3d')
            if rc != 0:
                self._recon_log_queue.put(('__done__', '3D reconstruction', rc))
                return
            rc = _run_step([
                sys.executable, 'make_3d_plot.py',
                '--input', proc_csv,
                '--output', html_out,
            ], 'make_3d_plot')
            self._recon_log_queue.put(('__done__', '3D reconstruction', rc))
            if rc == 0:
                import webbrowser
                webbrowser.open(Path(html_out).resolve().as_uri())

        threading.Thread(target=_worker, daemon=True).start()
        self.root.after(100, self._poll_recon_log)


# ══════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    root.update_idletasks()
    sh = root.winfo_screenheight()

    # パネルの canvas 高さを画面サイズから動的に決定
    # 固定オーバーヘッド: タイトルバー28 + navbar35 + notebook上部30 + パディング20 = 113
    # パネル非canvas部: label22 + nav30 + sync+load30 + パネル間パディング8 = 90
    # 2行分: 2 * (canvas + 90) + 113 <= sh
    panel_h = max(130, min(210, (sh - 113 - 2 * 90) // 2))
    global PANEL_H
    PANEL_H = panel_h

    root.resizable(True, True)
    SyncApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
