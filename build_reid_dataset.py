"""
build_reid_dataset.py
=====================
用途：把 PLEM 推估出來的行人軌跡，整理成「跨車行人重識別」的訓練資料集
      (RQ1, Learning-based 方法)。

流程對應計畫書第 7~12 頁的方法：
  1. 讀 inference_feedback_PLEM/ 裡每個 inference_GPS_{車}_{行人}.csv，
     只取 inside/outside==1 的列 (真的被偵測到的) 當作該車觀測到的軌跡。
  2. 把「車 Z 的某行人」與「車 Y 的某行人」兩兩配對：
        同編號 (例如 Z 的 P2 ↔ Y 的 P2)  -> label = 1 (正樣本，同一人)
        不同編號 (例如 Z 的 P2 ↔ Y 的 P3) -> label = 0 (負樣本，不同人)
  3. 取兩條軌跡的重疊時間區間 t_start=max(起點)、t_end=min(終點)，
     再用線性內插把 Y 的軌跡對齊到 Z 的時間點，得到等長序列。
  4. 算出 5 個特徵 f1~f5 (總長度差、平均速率差、平均逐點距離、
     方向一致性、加速度一致性)。
  5. 輸出 reid_features.csv (每一列 = 一對軌跡 + 標籤)。

用法：
    python build_reid_dataset.py  D:\CARLA_Experiments\20260324_004349
若不給參數，使用下方預設路徑。
"""

import os
import sys
import glob
import math
import numpy as np
import pandas as pd
from itertools import product

# 對齊後的軌跡至少要這麼多個點，才算得出有意義的特徵 (尤其方向、加速度)
MIN_LEN = 5


# ============================================================
# 讀檔與解析
# ============================================================
def parse_filename(filename):
    """inference_GPS_Z_P1.csv -> ('Z', 'P1')"""
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    return parts[2], parts[-1]          # car_id, ped_id


def load_track(csv_path):
    """回傳該車觀測到的軌跡 (只取 inside/outside==1)：frames, lats, lons (皆為 numpy 陣列，依 frame 排序)。"""
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None

    if "inside/outside" in df.columns:
        df = df[df["inside/outside"] == 1]
    df = df.dropna(subset=["predict_lat", "predict_lon", "frame"])
    df = df.sort_values("frame")

    if len(df) == 0:
        return None

    return (df["frame"].astype(float).values,
            df["predict_lat"].astype(float).values,
            df["predict_lon"].astype(float).values)


# ============================================================
# 經緯度 -> 區域平面公尺座標 (方便做向量運算)
# ============================================================
def to_local_xy(lats, lons, lat0, lon0):
    """以 (lat0, lon0) 為原點，把經緯度換成東(x)、北(y) 公尺座標。"""
    x_east = (lons - lon0) * 111320.0 * math.cos(math.radians(lat0))
    y_north = (lats - lat0) * 111320.0
    return np.column_stack([x_east, y_north])      # shape (n, 2)


def path_length(xy):
    """一條軌跡的幾何總長度 (公尺)。"""
    if len(xy) < 2:
        return 0.0
    seg = np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))
    return float(seg.sum())


# ============================================================
# 軌跡對齊：取重疊區間 + 線性內插
# ============================================================
def align_pair(track_v, track_u):
    """
    以 track_v (車 Z) 的時間點為基準，把 track_u (車 Y) 內插對齊。
    回傳 (v_lat, v_lon, u_lat, u_lon, ref_frames)，皆為等長陣列；若無法對齊則回傳 None。
    """
    fv, vlat, vlon = track_v
    fu, ulat, ulon = track_u

    t_start = max(fv.min(), fu.min())     # 起點取較晚的
    t_end = min(fv.max(), fu.max())       # 終點取較早的
    if t_start >= t_end:
        return None                       # 沒有時間重疊

    # 基準時間點 = 車 Z 落在重疊區間內的 frame
    mask = (fv >= t_start) & (fv <= t_end)
    ref_frames = fv[mask]
    if len(ref_frames) < MIN_LEN:
        return None                       # 重疊太短，組不成軌跡

    v_lat = vlat[mask]
    v_lon = vlon[mask]

    # 車 Y 用線性內插對齊到 ref_frames (區間在 Y 範圍內，屬內插非外推)
    u_lat = np.interp(ref_frames, fu, ulat)
    u_lon = np.interp(ref_frames, fu, ulon)

    return v_lat, v_lon, u_lat, u_lon, ref_frames


# ============================================================
# 計算 5 個特徵 (計畫書第 11~12 頁)
# ============================================================
def compute_features(v_lat, v_lon, u_lat, u_lon, ref_frames):
    # 用兩條軌跡所有點的平均位置當投影原點
    lat0 = float(np.concatenate([v_lat, u_lat]).mean())
    lon0 = float(np.concatenate([v_lon, u_lon]).mean())
    V = to_local_xy(v_lat, v_lon, lat0, lon0)     # (n,2)
    U = to_local_xy(u_lat, u_lon, lat0, lon0)
    n = len(V)

    # f1 總長度差
    len_v, len_u = path_length(V), path_length(U)
    f1 = abs(len_v - len_u)

    # f2 平均速率差 (對齊後兩者時間跨度相同，分母一致)
    span = ref_frames[-1] - ref_frames[0]
    span = span if span > 0 else 1.0
    f2 = abs(len_v / span - len_u / span)

    # f3 平均逐點距離
    ptdist = np.sqrt(((V - U) ** 2).sum(axis=1))
    f3 = float(ptdist.mean())

    # f4 方向一致性 (逐段位移向量的餘弦相似度平均，值域 -1~1)
    dV, dU = np.diff(V, axis=0), np.diff(U, axis=0)        # (n-1,2)
    dot = (dV * dU).sum(axis=1)
    nv = np.sqrt((dV ** 2).sum(axis=1))
    nu = np.sqrt((dU ** 2).sum(axis=1))
    denom = nv * nu
    cossim = np.where(denom > 1e-9, dot / np.where(denom > 1e-9, denom, 1.0), 0.0)
    f4 = float(cossim.mean()) if len(cossim) > 0 else 0.0

    # f5 加速度一致性 (加速度 = 連續位移向量的差分)
    if n >= 3:
        aV, aU = np.diff(dV, axis=0), np.diff(dU, axis=0)  # (n-2,2)
        f5 = float(np.sqrt(((aV - aU) ** 2).sum(axis=1)).mean())
    else:
        f5 = 0.0

    return f1, f2, f3, f4, f5


# ============================================================
# 主流程
# ============================================================
def build_dataset(base_folder):
    input_folder = os.path.join(base_folder, "inference_feedback_PLEM")
    files = sorted(glob.glob(os.path.join(input_folder, "inference_GPS_*.csv")))

    print("=" * 64)
    print("建立跨車行人重識別訓練資料集")
    print(f"資料夾: {input_folder}")
    print("=" * 64)

    if not files:
        print("❌ 找不到 inference_GPS_*.csv，請先跑完 run_inference_plem.py。")
        return

    # 依車輛分組： tracks[car_id][ped_id] = (frames, lats, lons)
    tracks = {}
    for path in files:
        car_id, ped_id = parse_filename(os.path.basename(path))
        t = load_track(path)
        if t is not None:
            tracks.setdefault(car_id, {})[ped_id] = t

    cars = sorted(tracks.keys())
    if len(cars) < 2:
        print(f"❌ 只找到車輛 {cars}，重識別至少需要兩台車的資料。")
        return

    car_v, car_u = cars[0], cars[1]       # 例如 Y、Z
    print(f"配對車輛：以 {car_v} 為基準時間軸，對齊 {car_u}\n")

    rows = []
    n_pos = n_neg = n_skip = 0

    # 兩台車的行人兩兩配對
    for ped_v, ped_u in product(tracks[car_v].keys(), tracks[car_u].keys()):
        label = 1 if ped_v == ped_u else 0
        aligned = align_pair(tracks[car_v][ped_v], tracks[car_u][ped_u])

        if aligned is None:
            n_skip += 1
            continue

        v_lat, v_lon, u_lat, u_lon, ref_frames = aligned
        f1, f2, f3, f4, f5 = compute_features(v_lat, v_lon, u_lat, u_lon, ref_frames)

        rows.append({
            "pair_id": f"{car_v}_{ped_v}__{car_u}_{ped_u}",
            "car_v": car_v, "ped_v": ped_v,
            "car_u": car_u, "ped_u": ped_u,
            "n_points": len(ref_frames),
            "overlap_start": int(ref_frames[0]),
            "overlap_end": int(ref_frames[-1]),
            "f1_len_diff": f1,
            "f2_speed_diff": f2,
            "f3_mean_dist": f3,
            "f4_dir_consistency": f4,
            "f5_acc_consistency": f5,
            "label": label,
        })
        if label == 1:
            n_pos += 1
        else:
            n_neg += 1

    if not rows:
        print("⚠️ 沒有任何一對軌跡的重疊長度達標，無法產生特徵。")
        print(f"   (目前每對至少需要重疊 {MIN_LEN} 幀；可調小 MIN_LEN，或讓兩車視野更交疊。)")
        return

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(base_folder, "reid_features.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("每對軌跡的特徵：")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(out_df[["pair_id", "n_points", "f1_len_diff", "f2_speed_diff",
                      "f3_mean_dist", "f4_dir_consistency", "f5_acc_consistency", "label"]]
              .round(4).to_string(index=False))

    print("\n" + "-" * 64)
    print("摘要：")
    print(f"  產生樣本對: {len(rows)}  (正樣本 {n_pos} 對、負樣本 {n_neg} 對)")
    print(f"  因重疊不足 {MIN_LEN} 幀而略過: {n_skip} 對")
    print(f"  已輸出: {out_path}")
    print("-" * 64)
    if len(rows) < 50:
        print("提醒：樣本數還很少，先用這份確認特徵算得對；之後讓 auto_pipeline 多跑幾輪，")
        print("      把多個實驗資料夾的 reid_features.csv 合併起來，才夠訓練分類器。")
    print("=" * 64)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        base_folder = r"D:\CARLA_Experiments\20260708_201641"   # 記得換成你的實驗資料夾

    build_dataset(base_folder)
