"""
probe_walkable.py
=================
在斑馬線附近逐格嘗試生成行人，畫出「哪裡可以生成 / 站得住」的文字地圖，
用來挑選 FIXED_ROUTES 的路線座標。只需在 CARLA 開著時執行一次。

用法：
    python probe_walkable.py
    python probe_walkable.py -64 -56 -14 17      # 自訂 x_min x_max y_min y_max

圖例：
    .  可以生成行人
    #  生成失敗 (有建物、柱子、牆、車等擋住)
    數字 1/2/3  目前 FIXED_ROUTES 中 P1/P2/P3 經過的格子 (該格可生成)
    !          路線經過、但該格無法生成 (需調整)
"""
import sys
import math
import carla

X_MIN, X_MAX, Y_MIN, Y_MAX = -64.0, -56.0, -14.0, 17.0
STEP = 0.5
TOWN = "Town05_Opt"

# 與 Precise_Vehicle_Placement_Random_v2.py 的 FIXED_ROUTES 保持一致 (只取 x 與 y 範圍)
ROUTES = {
    "1": [(-61.4, -10.5), (-61.4, 12.5)],
    "2": [(-60.4, 15.0), (-60.4, -8.5)],
    "3": [(-59.4, -10.5), (-59.4, 11.5)],
}


def on_route(x, y):
    for k, wps in ROUTES.items():
        for (x0, y0), (x1, y1) in zip(wps[:-1], wps[1:]):
            # 點到線段距離
            dx, dy = x1 - x0, y1 - y0
            t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)))
            if math.hypot(x - (x0 + t * dx), y - (y0 + t * dy)) <= STEP / 2:
                return k
    return None


def main():
    global X_MIN, X_MAX, Y_MIN, Y_MAX
    if len(sys.argv) == 5:
        X_MIN, X_MAX, Y_MIN, Y_MAX = map(float, sys.argv[1:5])

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    if not world.get_map().name.endswith(TOWN):
        print(f"載入地圖 {TOWN} ...")
        world = client.load_world(TOWN)

    # 用非同步模式探測 (try_spawn 不需要 tick)
    settings = world.get_settings()
    if settings.synchronous_mode:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

    bp = world.get_blueprint_library().find("walker.pedestrian.0001")

    xs = [X_MIN + i * STEP for i in range(int((X_MAX - X_MIN) / STEP) + 1)]
    ys = [Y_MIN + j * STEP for j in range(int((Y_MAX - Y_MIN) / STEP) + 1)]
    grid = {}
    print(f"探測 {len(xs)} x {len(ys)} = {len(xs) * len(ys)} 個點，請稍候...")
    for y in ys:
        for x in xs:
            a = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=x, y=y, z=1.0)))
            grid[(x, y)] = a is not None
            if a is not None:
                a.destroy()

    # 輸出：y 由大到小往下排 (列)，x 由小到大往右排 (欄)
    print("\n      x→ " + "".join(f"{x:<8.1f}" if i % 4 == 0 else "" for i, x in enumerate(xs)))
    bad = []
    for y in reversed(ys):
        row = ""
        for x in xs:
            k = on_route(x, y)
            ok = grid[(x, y)]
            if k:
                row += k if ok else "!"
                if not ok:
                    bad.append((k, x, y))
            else:
                row += "." if ok else "#"
        print(f"y={y:6.1f}  {row}")

    print("\n圖例: . 可生成   # 擋住   1/2/3 路線經過(可生成)   ! 路線經過但擋住")
    if bad:
        print("\n⚠️ 以下路線格子被擋住，建議調整 FIXED_ROUTES：")
        for k, x, y in bad:
            print(f"   P{k}: ({x:.1f}, {y:.1f})")
    else:
        print("\n✅ 目前三條路線經過的格子都可以生成行人。")


if __name__ == "__main__":
    main()