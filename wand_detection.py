"""
キャリブレーションワンド上の球体マーカー検出パイプライン

旧版の問題:
  - 固定輝度閾値 (bright_th=190) が距離・照明で合わなくなる
  - 面積フィルタ (area_range) が解像度・距離依存

改善点:
  1. 局所最大値 (local maxima) ベース検出
     → 輝度閾値不要。ROI内の「最も明るい点」を自動抽出
  2. DoG (Difference of Gaussians) でマーカーらしさを強調
     → 背景がぼんやり明るい場合でも点状の光沢を拾える
  3. NMS (Non-Maximum Suppression) で密集した重複候補を除去
  4. RANSAC 直線フィット + 等間隔制約は従来通り
"""

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter


# ── 1. 局所最大値ベースの球体候補検出 ────────────────
def detect_bright_blobs(img, x_search, y_search,
                         n_candidates: int = 30,
                         min_dist: int = 8,
                         # 旧パラメータ（互換性のため受け取るが使用しない）
                         bright_th=None, area_range=None):
    """
    ROI内で「局所的に明るい点」を候補として返す。
    グローバル輝度閾値を使わないため、照明・距離の変化に頑健。

    Parameters
    ----------
    n_candidates : 最終的に返す候補点の最大数
    min_dist     : 候補点間の最小距離 [px]（NMS の抑制半径）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x0, x1 = int(x_search[0]), int(x_search[1])
    y0, y1 = int(y_search[0]), int(y_search[1])
    crop = gray[y0:y1, x0:x1]

    # DoG: 小スケールの輝点を強調（背景のなだらかな明るさを除去）
    s1, s2 = 1.5, 4.0
    dog = gaussian_filter(crop, s1) - gaussian_filter(crop, s2)
    dog = np.clip(dog, 0, None)   # 暗くなる方向は無視

    # 局所最大値マスク
    footprint_size = max(3, min_dist // 2 * 2 + 1)
    local_max_mask = (dog == maximum_filter(dog, size=footprint_size))

    # 上位 p% の輝度のみ残す（ノイズ抑制）
    thresh = np.percentile(dog[dog > 0], 80) if dog.max() > 0 else 0
    local_max_mask &= dog >= thresh

    ys, xs = np.where(local_max_mask)
    if len(ys) == 0:
        return []

    values = dog[ys, xs]
    order = np.argsort(values)[::-1]   # 明るい順
    ys, xs, values = ys[order], xs[order], values[order]

    # NMS: min_dist より近い点は除外
    selected: list[tuple[float, float, float]] = []
    used = np.zeros(len(ys), dtype=bool)
    for i in range(len(ys)):
        if used[i]:
            continue
        cy, cx, cv_ = ys[i], xs[i], values[i]
        selected.append((float(cx + x0), float(cy + y0), float(cv_)))
        # 近傍を抑制
        dists = np.hypot(xs - cx, ys - cy)
        used |= dists < min_dist
        if len(selected) >= n_candidates:
            break

    selected.sort(key=lambda t: t[1])   # y 座標順
    return selected


# ── 2. RANSAC 直線推定 ───────────────────────────────
def fit_line_from_points(points_xy, ransac_iters=500, inlier_dist=5.0):
    """x = a*y + b の直線を RANSAC でロバスト推定"""
    pts = np.array(points_xy, dtype=np.float64)
    if len(pts) < 2:
        return None

    rng = np.random.default_rng(0)
    best_count, best_inliers, best_ab = 0, None, None

    for _ in range(ransac_iters):
        idx = rng.choice(len(pts), 2, replace=False)
        (x1_, y1_), (x2_, y2_) = pts[idx]
        if y1_ == y2_:
            continue
        a = (x2_ - x1_) / (y2_ - y1_)
        b = x1_ - a * y1_

        pred_x = a * pts[:, 1] + b
        dist    = np.abs(pts[:, 0] - pred_x)
        inliers = dist < inlier_dist

        if inliers.sum() > best_count:
            best_count   = inliers.sum()
            best_inliers = inliers
            best_ab      = (a, b)

    if best_ab is None or best_count < 2:
        return None

    ip = pts[best_inliers]
    A  = np.vstack([ip[:, 1], np.ones(len(ip))]).T
    a, b = np.linalg.lstsq(A, ip[:, 0], rcond=None)[0]
    return (a, b)


# ── 直線近傍に絞る ────────────────────────────────────
def filter_near_line(blobs, a, b, dist_th=6.0):
    return [(gx, gy, v, abs(gx - (a * gy + b)))
            for gx, gy, v in blobs
            if abs(gx - (a * gy + b)) <= dist_th]


# ── 3. 等間隔フィット（誤検出除去・補完）────────────────
def fit_equal_spacing(candidates, expected_count=4,
                       max_missing_between=2, tol=8.0, min_spacing=20.0):
    """
    候補点の y 座標から等間隔の格子を推定し、
    外れ値を除去・見逃しを補完する。
    """
    ys = sorted(set(c[1] for c in candidates))
    if len(ys) < 2:
        return None

    best = None

    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            diff = ys[j] - ys[i]
            for k in range(1, max_missing_between + 2):
                spacing = diff / k
                if spacing < min_spacing:
                    continue

                offsets = [(y - ys[i]) / spacing for y in ys]
                rounded = [round(o) for o in offsets]
                near    = [abs(o - r) * spacing <= tol
                           for o, r in zip(offsets, rounded)]

                used_slots = [r for r, n in zip(rounded, near) if n]
                if len(used_slots) != len(set(used_slots)):
                    continue

                count = sum(near)
                if (best is None or count > best[0]
                        or (count == best[0] and spacing > best[1])):
                    best = (count, spacing, ys[i])

    if best is None or best[0] < 2:
        return None

    _, spacing, base_y = best
    inlier_ys = [y for y in ys
                 if abs(round((y - base_y) / spacing)
                        - (y - base_y) / spacing) * spacing <= tol]
    offsets_arr = np.round([(y - base_y) / spacing for y in inlier_ys])
    A = np.vstack([offsets_arr, np.ones(len(offsets_arr))]).T
    spacing_fit, base_fit = np.linalg.lstsq(
        A, np.array(inlier_ys), rcond=None)[0]

    k_min   = int(np.floor((min(inlier_ys) - base_fit) / spacing_fit)) - 1
    all_g   = [base_fit + k * spacing_fit
               for k in range(k_min, k_min + expected_count + 3)]
    lo = min(ys) - spacing_fit * 0.6
    hi = max(ys) + spacing_fit * 0.6
    grid_y  = [y for y in all_g if lo <= y <= hi]

    return {'spacing': spacing_fit, 'grid_y': grid_y}


# ── メイン API ────────────────────────────────────────
def detect_wand_markers(image, x_search=None, y_search=None,
                         expected_count: int = 4,
                         n_candidates: int  = 30,
                         min_dist: int      = 8,
                         min_spacing: float = 20.0,
                         inlier_dist: float = 5.0,
                         # 旧パラメータ互換（無視）
                         bright_th=None, area_range=None):
    """
    image: BGR numpy 配列 or ファイルパス文字列
    戻り値: (points, line_ab)
      points  : [(x, y), ...] 検出マーカー座標（画像座標系）
      line_ab : (a, b)  x = a*y + b  / None なら検出失敗
    """
    img = cv2.imread(image) if isinstance(image, str) else image
    h, w = img.shape[:2]

    if x_search is None:
        x_search = (0, w)
    if y_search is None:
        y_search = (0, h)

    # STEP 1: 局所最大値で候補抽出
    blobs = detect_bright_blobs(img, x_search, y_search,
                                 n_candidates=n_candidates,
                                 min_dist=min_dist)
    if len(blobs) < 2:
        return [], None

    # STEP 2: RANSAC 直線
    line = fit_line_from_points([(x, y) for x, y, _ in blobs],
                                  inlier_dist=inlier_dist)
    if line is None:
        return [], None
    a, b = line

    # STEP 3: 直線近傍に絞り込む
    near = filter_near_line(blobs, a, b, dist_th=inlier_dist * 1.5)
    if not near:
        return [], (a, b)

    # STEP 4: 等間隔フィット
    fit = fit_equal_spacing(near, expected_count=expected_count,
                             min_spacing=min_spacing)
    if fit is None:
        pts = [(gx, gy) for gx, gy, _, _ in near[:expected_count]]
    else:
        pts = [(a * gy + b, gy) for gy in fit['grid_y']]

    return pts, (a, b)
