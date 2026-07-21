"""
build_reid_dataset.py  (擴充版：ML 特徵 f1~f5 + 傳統方法 ED / DTW / LCSS)
=======================================================================
把 PLEM 推估出的行人軌跡兩兩配對、時間對齊後，同時算出：
  - 機器學習法要用的 5 維特徵 f1~f5
  - 傳統軌跡相似度方法的分數：ED、DTW、LCSS  (計畫書 RQ1 的對照組)
輸出成含 label 的 reid_features.csv，供之後「ML vs 傳統方法」比較。

用法：
    python build_reid_dataset.py  D:\CARLA_Experiments\20260324_004349
"""

import os
import sys
import glob
import math
import numpy as np
import pandas as pd
from itertools import product

MIN_LEN = 5          # 對齊後至少幾個點才算特徵
LCSS_EPS = 3.0       # LCSS 判定「兩點算匹配」的距離門檻(公尺)；可依場景調整


# ============================================================
# 讀檔 / 解析 / 座標轉換
# ============================================================
def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    return parts[2], parts[-1]          # car_id, ped_id


def load_track(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
    if "inside/outside" in df.columns:
        df = df[df["inside/outside"] == 1]
    df = df.dropna(subset=["predict_lat", "predict_lon", "frame"]).sort_values("frame")
    if len(df) == 0:
        return None
    return (df["frame"].astype(float).values,
            df["predict_lat"].astype(float).values,
            df["predict_lon"].astype(float).values)


def to_local_xy(lats, lons, lat0, lon0):
    x_east = (lons - lon0) * 111320.0 * math.cos(math.radians(lat0))
    y_north = (lats - lat0) * 111320.0
    return np.column_stack([x_east, y_north])


def path_length(xy):
    if len(xy) < 2:
        return 0.0
    return float(np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1)).sum())


# ============================================================
# 時間對齊 (取重疊區間 + 線性內插)
# ============================================================
def align_pair(track_v, track_u):
    fv, vlat, vlon = track_v
    fu, ulat, ulon = track_u
    t_start = max(fv.min(), fu.min())
    t_end = min(fv.max(), fu.max())
    if t_start >= t_end:
        return None
    mask = (fv >= t_start) & (fv <= t_end)
    ref_frames = fv[mask]
    if len(ref_frames) < MIN_LEN:
        return None
    v_lat, v_lon = vlat[mask], vlon[mask]
    u_lat = np.interp(ref_frames, fu, ulat)
    u_lon = np.interp(ref_frames, fu, ulon)
    return v_lat, v_lon, u_lat, u_lon, ref_frames


# ============================================================
# 傳統軌跡相似度方法 (對齊後的公尺座標序列 V、U)
# ============================================================
def ed_distance(V, U):
    """Euclidean Distance：逐點歐氏距離的平均 (公尺，越小越像)。"""
    return float(np.sqrt(((V - U) ** 2).sum(axis=1)).mean())


def dtw_distance(V, U):
    """Dynamic Time Warping：允許時間軸彈性對齊的最小累積距離，
       回傳除以序列長度的『每步平均』DTW 距離 (越小越像)。"""
    n, m = len(V), len(U)
    INF = float("inf")
    dp = np.full((n + 1, m + 1), INF)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        vi = V[i - 1]
        for j in range(1, m + 1):
            cost = math.sqrt(((vi - U[j - 1]) ** 2).sum())
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(n, m))


def lcss_similarity(V, U, eps=LCSS_EPS):
    """Longest Common Subsequence：兩軌跡在 eps 公尺內可視為匹配的最長共同子序列，
       除以較短長度正規化到 0~1 (越大越像)。"""
    n, m = len(V), len(U)
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        vi = V[i - 1]
        for j in range(1, m + 1):
            if math.sqrt(((vi - U[j - 1]) ** 2).sum()) < eps:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return float(dp[n, m] / min(n, m))


# ============================================================
# 計算所有分數 (ML 特徵 f1~f5 + 傳統 ED/DTW/LCSS)
# ============================================================
def compute_all(v_lat, v_lon, u_lat, u_lon, ref_frames):
    lat0 = float(np.concatenate([v_lat, u_lat]).mean())
    lon0 = float(np.concatenate([v_lon, u_lon]).mean())
    V = to_local_xy(v_lat, v_lon, lat0, lon0)
    U = to_local_xy(u_lat, u_lon, lat0, lon0)
    n = len(V)

    # --- ML 特徵 f1~f5 ---
    len_v, len_u = path_length(V), path_length(U)
    f1 = abs(len_v - len_u)

    span = ref_frames[-1] - ref_frames[0]
    span = span if span > 0 else 1.0
    f2 = abs(len_v / span - len_u / span)

    ptdist = np.sqrt(((V - U) ** 2).sum(axis=1))
    f3 = float(ptdist.mean())

    dV, dU = np.diff(V, axis=0), np.diff(U, axis=0)
    dot = (dV * dU).sum(axis=1)
    nv = np.sqrt((dV ** 2).sum(axis=1))
    nu = np.sqrt((dU ** 2).sum(axis=1))
    denom = nv * nu
    cossim = np.where(denom > 1e-9, dot / np.where(denom > 1e-9, denom, 1.0), 0.0)
    f4 = float(cossim.mean()) if len(cossim) > 0 else 0.0

    if n >= 3:
        aV, aU = np.diff(dV, axis=0), np.diff(dU, axis=0)
        f5 = float(np.sqrt(((aV - aU) ** 2).sum(axis=1)).mean())
    else:
        f5 = 0.0

    # --- 傳統方法 ---
    ed = ed_distance(V, U)
    dtw = dtw_distance(V, U)
    lcss = lcss_similarity(V, U)

    return {
        "f1_len_diff": f1, "f2_speed_diff": f2, "f3_mean_dist": f3,
        "f4_dir_consistency": f4, "f5_acc_consistency": f5,
        "ed_dist": ed, "dtw_dist": dtw, "lcss_sim": lcss,
    }


# ============================================================
# 主流程
# ============================================================
def build_dataset(base_folder):
    input_folder = os.path.join(base_folder, "inference_feedback_PLEM")
    files = sorted(glob.glob(os.path.join(input_folder, "inference_GPS_*.csv")))

    print("=" * 66)
    print("建立跨車行人重識別訓練資料集 (ML 特徵 + 傳統 ED/DTW/LCSS)")
    print(f"資料夾: {input_folder}")
    print("=" * 66)

    if not files:
        print("❌ 找不到 inference_GPS_*.csv，請先跑完 run_inference_plem.py。")
        return

    tracks = {}
    for path in files:
        car_id, ped_id = parse_filename(os.path.basename(path))
        t = load_track(path)
        if t is not None:
            tracks.setdefault(car_id, {})[ped_id] = t

    cars = sorted(tracks.keys())
    if len(cars) < 2:
        print(f"❌ 只找到車輛 {cars}，重識別至少需要兩台車。")
        return

    car_v, car_u = cars[0], cars[1]
    print(f"配對車輛：以 {car_v} 為基準時間軸，對齊 {car_u}\n")

    rows = []
    n_pos = n_neg = n_skip = 0
    for ped_v, ped_u in product(tracks[car_v].keys(), tracks[car_u].keys()):
        label = 1 if ped_v == ped_u else 0
        aligned = align_pair(tracks[car_v][ped_v], tracks[car_u][ped_u])
        if aligned is None:
            n_skip += 1
            continue
        v_lat, v_lon, u_lat, u_lon, ref_frames = aligned
        scores = compute_all(v_lat, v_lon, u_lat, u_lon, ref_frames)
        row = {
            "pair_id": f"{car_v}_{ped_v}__{car_u}_{ped_u}",
            "car_v": car_v, "ped_v": ped_v, "car_u": car_u, "ped_u": ped_u,
            "n_points": len(ref_frames),
            "overlap_start": int(ref_frames[0]), "overlap_end": int(ref_frames[-1]),
        }
        row.update(scores)
        row["label"] = label
        rows.append(row)
        n_pos += (label == 1)
        n_neg += (label == 0)

    if not rows:
        print(f"⚠️ 沒有任何一對軌跡重疊達 {MIN_LEN} 幀，無法產生特徵。")
        return

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(base_folder, "reid_features.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("每對軌跡 (ML 特徵 + 傳統方法) 摘要：")
    show = ["pair_id", "n_points", "f3_mean_dist", "f4_dir_consistency",
            "ed_dist", "dtw_dist", "lcss_sim", "label"]
    with pd.option_context("display.max_columns", None, "display.width", 130):
        print(out_df[show].round(4).to_string(index=False))

    print("\n" + "-" * 66)
    print(f"  產生樣本對: {len(rows)}  (正 {n_pos} / 負 {n_neg})，因重疊不足略過: {n_skip}")
    print(f"  已輸出: {out_path}")
    print("  欄位: f1~f5 (ML) + ed_dist / dtw_dist / lcss_sim (傳統) + label")
    print("=" * 66)


if __name__ == "__main__":
    base_folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\CARLA_Experiments\20260324_004349"
    build_dataset(base_folder)
