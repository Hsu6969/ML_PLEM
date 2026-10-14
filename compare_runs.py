"""
compare_runs.py
===============
比較兩次實驗 (例如 1280 存檔 vs 6144 存檔) 在「上游」是否真的有差：
不看最後的分類 F1 (樣本太少時雜訊很大)，而是逐場景比較每一幀的
  - YOLO 有配對到行人的幀數 (inside/outside == 1)
  - BBox 高度
  - PLEM 推估位置誤差 Δd (GPS_difference，公尺)
兩次實驗的同一場景第 k 輪使用相同 route-seed，行人軌跡相同，因此可以直接比較。

用法：
    python compare_runs.py  D:\\CARLA_Experiments\\run_A  D:\\CARLA_Experiments\\run_B
    (第一個資料夾當 A，第二個當 B；例如 A = 1280、B = 6144)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd


def collect(run_folder):
    """回傳 {(場景, 車, 行人): DataFrame}，只取各場景第一個 (時間最早的) 輪次。"""
    out = {}
    for sc_dir in sorted(glob.glob(os.path.join(run_folder, "*"))):
        if not os.path.isdir(sc_dir):
            continue
        rounds = sorted(d for d in glob.glob(os.path.join(sc_dir, "*"))
                        if os.path.isdir(os.path.join(d, "inference_feedback_PLEM")))
        if not rounds:
            continue
        sc = os.path.basename(sc_dir)
        for f in glob.glob(os.path.join(rounds[0], "inference_feedback_PLEM", "inference_GPS_*.csv")):
            parts = os.path.splitext(os.path.basename(f))[0].split("_")
            car, ped = parts[2], parts[-1]
            df = pd.read_csv(f).sort_values("frame").reset_index(drop=True)
            df["bbox_h"] = df["y_rightbottom"] - df["y_left_up"]
            out[(sc, car, ped)] = df
    return out


def summarize(df):
    det = df[df["inside/outside"] == 1]
    return {
        "frames": len(df),
        "detected": len(det),
        "dd_mean": det["GPS_difference"].mean() if len(det) else np.nan,
        "dd_median": det["GPS_difference"].median() if len(det) else np.nan,
        "h_mean": det["bbox_h"].mean() if len(det) else np.nan,
    }


def main(run_a, run_b):
    A, B = collect(run_a), collect(run_b)
    keys = sorted(set(A) & set(B))
    print("=" * 92)
    print(f"A = {run_a}\nB = {run_b}")
    print("=" * 92)
    if not keys:
        print("❌ 兩邊找不到相同的 (場景, 車, 行人)，請確認兩個 run 資料夾都有跑完。")
        return

    rows = []
    for k in keys:
        a, b = summarize(A[k]), summarize(B[k])
        # 兩邊都偵測到的幀：逐幀比較 (以行人檔內的順序對齊，同 route-seed 時軌跡一致)
        n = min(len(A[k]), len(B[k]))
        da, db = A[k].iloc[:n], B[k].iloc[:n]
        both = (da["inside/outside"].values == 1) & (db["inside/outside"].values == 1)
        dd_gap = np.abs(da["GPS_difference"].values[both] - db["GPS_difference"].values[both])
        rows.append({"scene": k[0], "car": k[1], "ped": k[2],
                     "A_det": a["detected"], "B_det": b["detected"], "frames": a["frames"],
                     "A_dd": a["dd_mean"], "B_dd": b["dd_mean"],
                     "A_h": a["h_mean"], "B_h": b["h_mean"],
                     "both": int(both.sum()),
                     "dd_gap": dd_gap.mean() if len(dd_gap) else np.nan})

    r = pd.DataFrame(rows)
    print(f"{'場景':<5}{'車':<3}{'行人':<5}{'偵測幀 A/B':>12}{'Δd平均 A':>10}{'Δd平均 B':>10}"
          f"{'框高 A':>9}{'框高 B':>9}{'同幀Δd差':>10}")
    for _, x in r.iterrows():
        print(f"{x['scene']:<5}{x['car']:<3}{x['ped']:<5}{int(x['A_det']):>6}/{int(x['B_det']):<5}"
              f"{x['A_dd']:>10.2f}{x['B_dd']:>10.2f}{x['A_h']:>9.3f}{x['B_h']:>9.3f}{x['dd_gap']:>10.2f}")

    print("-" * 92)
    print("整體 (所有場景/車/行人)：")
    print(f"  偵測到的幀數      A {int(r['A_det'].sum())}   B {int(r['B_det'].sum())}")
    print(f"  Δd 平均 (公尺)     A {np.nanmean(r['A_dd']):.2f}   B {np.nanmean(r['B_dd']):.2f}")
    print(f"  BBox 高度平均      A {np.nanmean(r['A_h']):.3f}  B {np.nanmean(r['B_h']):.3f}")
    print(f"  同一幀兩邊 Δd 平均差 {np.nanmean(r['dd_gap']):.2f} m")
    print("  判讀：偵測幀數與 Δd 都接近 -> 兩種解析度產生的資料基本一致，F1 的差異來自樣本太少；")
    print("        若某一邊偵測幀數明顯較少或 Δd 明顯較大 -> 解析度確實影響上游，需要進一步檢查。")
    print("=" * 92)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python compare_runs.py <run資料夾A> <run資料夾B>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
