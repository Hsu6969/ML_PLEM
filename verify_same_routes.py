"""
verify_same_routes.py
=====================
確認「同一個路徑種子」在 AB / CD / EF / GH 四組場景中，行人的真實軌跡是否相同。
讀每一輪的 scenario.txt 取得 route_seed，再比較 pedestrians.csv 裡 P1~P3 的真實經緯度。

用法：
    python verify_same_routes.py  D:\\CARLA_Experiments\\run_xxxx
    (可一次給多個 run 資料夾，例如舊 run + 接續跑的新 run)
"""
import os
import sys
import glob
import math
import pandas as pd


def read_seed(round_dir):
    p = os.path.join(round_dir, "scenario.txt")
    if not os.path.exists(p):
        return None, None
    lines = open(p, encoding="utf-8").read().splitlines()
    scene = lines[0].strip() if lines else None
    seed = None
    for ln in lines:
        if ln.startswith("route_seed="):
            seed = int(ln.split("=")[1])
    return scene, seed


def to_m(lat1, lon1, lat2, lon2):
    """兩組經緯度的距離 (公尺，平面近似，區域很小時足夠精確)。"""
    dy = (lat2 - lat1) * 111320.0
    dx = (lon2 - lon1) * 111320.0 * math.cos(math.radians(lat1))
    return math.hypot(dx, dy)


def main(run_folders):
    # seed -> {scene: DataFrame}
    by_seed = {}
    for run in run_folders:
        for rd in glob.glob(os.path.join(run, "*", "*")):
            ped_csv = os.path.join(rd, "pedestrians.csv")
            if not os.path.exists(ped_csv):
                continue
            scene, seed = read_seed(rd)
            if seed is None:
                continue
            by_seed.setdefault(seed, {})[scene] = pd.read_csv(ped_csv).sort_values("frame").reset_index(drop=True)

    if not by_seed:
        print("❌ 找不到含 scenario.txt (route_seed) 與 pedestrians.csv 的輪次資料夾")
        return

    peds = ["P1", "P2", "P3"]
    print("=" * 78)
    print("同一路徑種子在各場景的行人真實軌跡比較 (數值 = 同一時刻位置差的最大值，公尺)")
    print("=" * 78)
    print(f"{'種子':<6}{'場景':<16}{'對齊方式':<10}" + "".join(f"{p:>8}" for p in peds) + f"{'幀數':>14}")

    worst = 0.0
    for seed in sorted(by_seed):
        scenes = by_seed[seed]
        if len(scenes) < 2:
            continue
        names = sorted(scenes)
        ref_name = names[0]
        ref = scenes[ref_name]
        for other in names[1:]:
            o = scenes[other]
            relative = ref["frame"].max() < 100000 and o["frame"].max() < 100000
            if relative:      # 新資料：frame 從 0 開始，直接用 frame 對齊
                m = ref.merge(o, on="frame", suffixes=("_a", "_b"))
                how = "frame"
            else:             # 舊資料：frame 是 CARLA 原始編號，改用順序對齊
                n = min(len(ref), len(o))
                a = ref.iloc[:n].add_suffix("_a")
                b = o.iloc[:n].add_suffix("_b")
                m = pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1)
                how = "順序"
            vals = []
            for p in peds:
                if f"{p}_lat_a" not in m or f"{p}_lat_b" not in m:
                    vals.append(float("nan"))
                    continue
                d = [to_m(r[f"{p}_lat_a"], r[f"{p}_lon_a"], r[f"{p}_lat_b"], r[f"{p}_lon_b"])
                     for _, r in m.iterrows()]
                vals.append(max(d) if d else float("nan"))
            worst = max([worst] + [v for v in vals if v == v])
            print(f"{seed:<6}{ref_name + ' vs ' + other:<16}{how:<10}"
                  + "".join(f"{v:>8.2f}" for v in vals)
                  + f"{str(len(ref)) + '/' + str(len(o)):>14}")

    print("-" * 78)
    print(f"所有比較中最大的位置差: {worst:.2f} m")
    if worst < 0.3:
        print("✅ 各場景的行人軌跡實質上相同，可以跨場景直接比對。")
    else:
        print("⚠️ 有種子的軌跡差異超過 0.3 m，請看上表是哪個種子、哪個場景、哪位行人。")
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python verify_same_routes.py <run資料夾> [更多 run 資料夾 ...]")
        sys.exit(1)
    main(sys.argv[1:])
