"""
generate_route_sets.py
======================
產生「行人軌跡設定集」：把每個路徑種子 (1 ~ N) 對應的三位行人路線寫成檔案，
讓每一輪用哪一組路線變成看得見、可以放進論文附錄的固定清單。

- 路線設定 (FIXED_ROUTES、JITTER) 直接從採集程式讀取，不需要手動同步兩份數字。
- 產生的結果與採集程式內 make_fixed_routes(seed) 完全相同，所以已經跑完的輪次不受影響。

輸出 (放在本程式同一資料夾)：
  route_sets.json : 採集程式讀取用
  route_sets.csv  : 給人看的表格 (每列一位行人)

用法：
    python generate_route_sets.py            # 預設產生 30 組
    python generate_route_sets.py 40         # 產生 40 組
"""
import os
import sys
import ast
import json
import random
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTOR = os.path.join(HERE, "Precise_Vehicle_Placement_Random_v2.py")


def read_constants(path, names):
    """用 ast 從採集程式讀出常數 (不執行該程式，避免連線 CARLA 或建立資料夾)。"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in names:
                out[t.id] = ast.literal_eval(node.value)
    missing = [n for n in names if n not in out]
    if missing:
        raise RuntimeError(f"在 {path} 找不到常數: {missing}")
    return out


def make_routes(seed, fixed_routes, jitter):
    """與採集程式 make_fixed_routes() 相同的演算法。"""
    rng = random.Random(seed)
    routes = []
    for r in fixed_routes:
        wps = [tuple(p) for p in r["waypoints"]]
        speed, delay = r["speed"], r["delay"]
        if seed > 0:
            dx = rng.uniform(-jitter["lane_x"], jitter["lane_x"])
            dy0 = rng.uniform(-jitter["end_y"], jitter["end_y"])
            dy1 = rng.uniform(-jitter["end_y"], jitter["end_y"])
            wps = [(x + dx, y) for x, y in wps]
            wps[0] = (wps[0][0], wps[0][1] + dy0)
            wps[-1] = (wps[-1][0], wps[-1][1] + dy1)
            speed = round(speed + rng.uniform(-jitter["speed"], jitter["speed"]), 2)
            delay = max(0.0, round(delay + rng.uniform(-jitter["delay"], jitter["delay"]), 2))
        routes.append({"id": r["id"], "bp": r["bp"],
                       "waypoints": [[round(x, 4), round(y, 4)] for x, y in wps],
                       "speed": speed, "delay": delay})
    return routes


def main(n):
    c = read_constants(COLLECTOR, ["FIXED_ROUTES", "JITTER"])
    sets = {str(seed): make_routes(seed, c["FIXED_ROUTES"], c["JITTER"]) for seed in range(1, n + 1)}

    with open(os.path.join(HERE, "route_sets.json"), "w", encoding="utf-8") as f:
        json.dump(sets, f, ensure_ascii=False, indent=1)

    with open(os.path.join(HERE, "route_sets.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["route_set", "pedestrian", "start_x", "start_y", "end_x", "end_y",
                    "length_m", "speed_mps", "delay_s"])
        for seed, routes in sets.items():
            for r in routes:
                (x0, y0), (x1, y1) = r["waypoints"][0], r["waypoints"][-1]
                L = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
                        for a, b in zip(r["waypoints"][:-1], r["waypoints"][1:]))
                w.writerow([seed, r["id"], x0, y0, x1, y1, round(L, 2), r["speed"], r["delay"]])

    print(f"✅ 已產生 {n} 組行人軌跡設定：")
    print(f"   {os.path.join(HERE, 'route_sets.json')}  (採集程式讀取用)")
    print(f"   {os.path.join(HERE, 'route_sets.csv')}   (可放進論文附錄)")
    print("\n前 3 組預覽：")
    for seed in list(sets)[:3]:
        print(f" 第 {seed} 組:")
        for r in sets[seed]:
            print(f"   {r['id']}: {r['waypoints'][0]} → {r['waypoints'][-1]}  "
                  f"speed={r['speed']}  delay={r['delay']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
