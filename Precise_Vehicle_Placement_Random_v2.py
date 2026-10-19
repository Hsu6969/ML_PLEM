import carla
import argparse
import os
import re
from queue import Queue
from queue import Empty
import random
import pandas as pd
import math
from haversine import haversine
import time
from datetime import datetime
import sys
import numpy as np
import cv2
import faulthandler

# ============================================================
# 全域資料容器
# ============================================================
image_count = 0
gnss_data_Z, gnss_data_Y = [], []
imu_data_Z, imu_data_Y = [], []
pedestrians_data_list = []
BASE_FRAME = None   # ★ 本輪起始的 CARLA frame；所有 frame 都減掉它，讓每輪從 0 開始

# ============================================================
# ★ 車輛場景設定 (程式內仍沿用 Z / Y 命名：代號第 1 個字母 = Z 車，第 2 個 = Y 車)
#   格式: (x, y, yaw)；實驗變因只有兩車位置，行人路徑四組完全相同
# ============================================================
DEFAULT_SCENARIO = "AB"
SCENARIOS = {
    "AB": {"Z": (-67.5, 2.75, 0),    "Y": (-67.5, 6.1, 0)},        # 南側同向並排
    "CD": {"Z": (-43.5, 19.1, -90),  "Y": (-47.0, 19.1, -90)},     # 右上方，朝 -y
    "EF": {"Z": (-33.25, -4.2, 180), "Y": (-33.25, -1.0, 180)},    # 對向，朝 -x
    "GH": {"Z": (-51.0, -14.8, 90),  "Y": (-54.75, -14.8, 90)},    # 左下方，朝 +y
}

# ============================================================
# ★ 固定行人路徑 (不使用 AI 控制器，改用 WalkerControl 直接走直線)
#   - AI 控制器遇到其他行人會自動避讓、停頓數秒，這是之前「走一半停下來」的原因；
#     直接控制則每個行人以固定速度沿固定路線走，不會互相避讓或停頓。
#   - 三條路線在 x 方向錯開 1.2 m (互不碰撞)，並以出發延遲錯開時間。
#   - waypoints: 依序經過的點 (x, y)；可加中間點做轉彎
#   - delay: 錄影開始後幾秒才出發；speed: 公尺/秒
# ============================================================
FIXED_ROUTES = [
    {"id": "P1", "bp": "walker.pedestrian.0001", "waypoints": [(-61.4, -10.5), (-61.4, 12.5)],
     "speed": 1.30, "delay": 0.0},  # 由下往上
    {"id": "P2", "bp": "walker.pedestrian.0002", "waypoints": [(-60.4, 15.0), (-60.4, -8.5)],
     "speed": 1.20, "delay": 0.0},  # 由上往下 (方向相反的負樣本)
    {"id": "P3", "bp": "walker.pedestrian.0003", "waypoints": [(-59.4, -10.5), (-59.4, 11.5)],
     "speed": 1.35, "delay": 0.5},  # ★ 困難案例：與 P1 同方向、幾乎同時出發、相隔約 2 m 並排走
]

# --route-seed > 0 時，每輪對固定路線加入小幅擾動，避免多輪資料完全相同；
# 同一個 route-seed 在四個場景產生「完全相同」的行人軌跡 (控制變因)。
JITTER = {"lane_x": 0.1, "end_y": 0.5, "speed": 0.1, "delay": 1.0}
ARRIVE_DIST = 0.4      # 距離航點多近算抵達 (m)
STUCK_SECONDS = 4.0    # 行進中連續這麼久位移 < 0.3 m 視為卡住 (已嘗試跳上台階仍失敗)
# ★ 台階輔助：直接控制的行人「往下走台階」沒問題，但「往上跨人行道台階」常會卡住，
#   偵測到前進受阻時自動做一次小跳躍跨上去 (同步模式下仍是確定性的，各場景結果一致)
STEP_CHECK_SEC = 0.4   # 每隔多久檢查一次前進量
STEP_MIN_RATIO = 0.35  # 實際位移 < 應走距離的 35% 視為被台階擋住
JUMP_COOLDOWN = 1.0    # 兩次跨步至少間隔幾秒
STEP_UP_HEIGHT = 0.25  # ★ 跨上台階時往上抬的高度 (m)；人行道台階約 0.15~0.2 m
STEP_UP_FORWARD = 0.35 # ★ 同時往前移的距離 (m)，約半步
SAVE_SIZE = 1280  
HANG_SECONDS = 120   # ★ 超過這麼久沒有推進模擬，就印出卡住位置並強制結束 (讓 auto_pipeline 重試)

# ============================================================
# 參數解析：輸出資料夾 (位置參數) + --scenario
#   例: python Precise_Vehicle_Placement_Random_v2.py --scenario AG D:\CARLA_Experiments\xxx
# ============================================================
_ap = argparse.ArgumentParser(description="CARLA 雙車行人資料採集")
_ap.add_argument("output", nargs="?", default=None, help="本輪輸出資料夾")
_ap.add_argument("--scenario", default=None, help=f"車輛場景代號: {', '.join(SCENARIOS)}")
_ap.add_argument("--route-seed", type=int, default=0,
                 help="行人路徑擾動種子：0 = 完全使用基準固定路徑；相同種子在各場景產生相同軌跡")
_ap.add_argument("--sync", action="store_true", default=True, help="Synchronous mode")
ARGS = _ap.parse_args()

SCENARIO = ARGS.scenario or DEFAULT_SCENARIO
if SCENARIO not in SCENARIOS:
    print(f"❌ 未知的場景代號 '{SCENARIO}'，可用: {', '.join(SCENARIOS)}")
    sys.exit(2)
SCN = SCENARIOS[SCENARIO]

if ARGS.output:
    output_path_ego = ARGS.output
    print(f"🔗 [自動化管線] 接收到指定資料夾路徑: {output_path_ego}")
else:
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path_ego = os.path.join('D:/CARLA_Experiments', current_time_str)
    print(f"⏰ [手動執行] 自行建立時間資料夾: {output_path_ego}")
print(f"🚗 本輪場景: {SCENARIO}")

print(f"📁 本次實驗數據將儲存於: {output_path_ego}")

ROUTE_SEED = ARGS.route_seed
print(f"🛣️ 行人路徑種子: {ROUTE_SEED} ({'基準固定路徑' if ROUTE_SEED == 0 else '固定路徑 + 小幅擾動'})")

if not os.path.exists(output_path_ego):
    os.makedirs(output_path_ego)

output_path_image_Z = os.path.join(output_path_ego, 'image_Z')
output_path_image_Y = os.path.join(output_path_ego, 'image_Y')
os.makedirs(output_path_image_Z, exist_ok=True)
os.makedirs(output_path_image_Y, exist_ok=True)

town = 'Town05_Opt'
weather = carla.WeatherParameters.ClearNoon

def make_fixed_routes(route_seed):
    """回傳本輪 3 位行人的路線；route_seed 相同 -> 路線完全相同 (與場景無關)。"""
    rng = random.Random(route_seed)
    routes = []
    for r in FIXED_ROUTES:
        wps = list(r["waypoints"])
        speed, delay = r["speed"], r["delay"]
        if route_seed > 0:
            dx = rng.uniform(-JITTER["lane_x"], JITTER["lane_x"])          # 整條路線平移 (保持車道錯開)
            dy0 = rng.uniform(-JITTER["end_y"], JITTER["end_y"])
            dy1 = rng.uniform(-JITTER["end_y"], JITTER["end_y"])
            wps = [(x + dx, y) for x, y in wps]
            wps[0] = (wps[0][0], wps[0][1] + dy0)
            wps[-1] = (wps[-1][0], wps[-1][1] + dy1)
            speed = round(speed + rng.uniform(-JITTER["speed"], JITTER["speed"]), 2)
            delay = max(0.0, round(delay + rng.uniform(-JITTER["delay"], JITTER["delay"]), 2))
        routes.append({"id": r["id"], "bp": r["bp"], "waypoints": wps, "speed": speed, "delay": delay})
    print("🚶 本輪行人固定路徑:")
    for r in routes:
        path = " → ".join(f"({x:.1f},{y:.1f})" for x, y in r["waypoints"])
        print(f"   {r['id']}: {path}  speed={r['speed']} m/s  delay={r['delay']} s")
    return routes

ROUTE_SET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_sets.json")

def load_route_set(route_seed):
    """★ 從行人軌跡設定集 (route_sets.json) 讀取第 route_seed 組路線；
       檔案不存在或沒有這一組時回傳 None，改用 make_fixed_routes() 即時產生。"""
    if route_seed <= 0 or not os.path.exists(ROUTE_SET_FILE):
        return None
    import json
    sets = json.load(open(ROUTE_SET_FILE, encoding="utf-8"))
    if str(route_seed) not in sets:
        print(f"⚠️ 軌跡設定集中沒有第 {route_seed} 組 (共 {len(sets)} 組)，改用種子即時產生")
        return None
    routes = []
    for r in sets[str(route_seed)]:
        routes.append({"id": r["id"], "bp": r["bp"],
                       "waypoints": [tuple(p) for p in r["waypoints"]],
                       "speed": r["speed"], "delay": r["delay"]})
    print(f"🚶 使用軌跡設定集第 {route_seed} 組:")
    for r in routes:
        path = " → ".join(f"({x:.1f},{y:.1f})" for x, y in r["waypoints"])
        print(f"   {r['id']}: {path}  speed={r['speed']} m/s  delay={r['delay']} s")
    return routes

def _destroy_peds(active):
    for p in active:
        try:
            if p.get("actor") is not None:
                p["actor"].destroy()
        except Exception:
            pass


def spawn_fixed_pedestrians(world, blueprint_library, routes):
    """在路線起點生成行人 (不加 AI 控制器)；任一人生成失敗就丟出錯誤。"""
    active = []
    for r in routes:
        bp = blueprint_library.find(r["bp"])
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'true')
        (x0, y0), (x1, y1) = r["waypoints"][0], r["waypoints"][1]
        yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))     # 一開始就面向前進方向
        actor, used = None, None
        L = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / L, (y1 - y0) / L               # 前進方向
        px, py = -uy, ux                                    # 左右方向
        # 依序嘗試：沿路線往前 0~4 m、左右偏移 0 / ±0.3 m、高度 1.0 / 1.5 m
        for fwd in [i * 0.5 for i in range(9)]:
            for lat in (0.0, 0.3, -0.3):
                sx, sy = x0 + ux * fwd + px * lat, y0 + uy * fwd + py * lat
                for z in (1.0, 1.5):
                    actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=sx, y=sy, z=z),
                                                                      carla.Rotation(yaw=yaw)))
                    if actor is not None:
                        used = (sx, sy)
                        break
                if actor: break
            if actor: break
        if actor is None:
            _destroy_peds(active)
            raise RuntimeError(f"行人 {r['id']} 在起點 ({x0:.1f},{y0:.1f}) 附近 4 m 內都無法生成；"
                               "請先執行 probe_walkable.py 查看可行走範圍，再調整 FIXED_ROUTES")
        if math.hypot(used[0] - x0, used[1] - y0) > 0.01:
            print(f"   ⚠️ {r['id']} 原起點 ({x0:.1f},{y0:.1f}) 被擋住，改從 ({used[0]:.1f},{used[1]:.1f}) 出發")
            r["waypoints"][0] = used
        active.append({"name": r["id"], "actor": actor, "waypoints": r["waypoints"],
                       "wp_idx": 1, "speed": r["speed"], "delay": r["delay"],
                       "is_finished": False, "last_pos": None, "still_since": None,
                       "chk_pos": None, "chk_time": None, "last_jump": -99.0, "n_jumps": 0})
    # 讓行人落地站穩
    for _ in range(10):
        world.tick()
    print(f"   ✓ 已生成 {len(active)} 位行人 (直接控制，不使用 AI 避讓)")
    return active


def stop_walker(p):
    p["actor"].apply_control(carla.WalkerControl(direction=carla.Vector3D(0, 0, 0),
                                                 speed=0.0, jump=False))


def update_walkers(active, sim_time):
    """每個 tick 呼叫：讓每位行人朝下一個航點以固定速度直線前進。"""
    for p in active:
        if p["is_finished"]:
            continue
        if sim_time < p["delay"]:
            stop_walker(p)                       # 還沒到出發時間，原地等待
            continue
        loc = p["actor"].get_location()
        tx, ty = p["waypoints"][p["wp_idx"]]
        dx, dy = tx - loc.x, ty - loc.y
        dist = math.hypot(dx, dy)
        if dist < ARRIVE_DIST:
            if p["wp_idx"] < len(p["waypoints"]) - 1:
                p["wp_idx"] += 1                 # 前往下一個航點
                continue
            stop_walker(p)
            p["is_finished"] = True
            print(f"🚦 行人 {p['name']} 已抵達終點，停止移動。")
            continue
        # ★ 台階輔助：前進量明顯不足 -> 把行人往前、往上微移一小步，模擬「跨上」台階
        #   (CARLA 的 jump 高度固定且很高，改用小幅位移，看起來像正常走上人行道)
        if p["chk_pos"] is None:
            p["chk_pos"], p["chk_time"] = (loc.x, loc.y), sim_time
        elif sim_time - p["chk_time"] >= STEP_CHECK_SEC:
            moved = math.hypot(loc.x - p["chk_pos"][0], loc.y - p["chk_pos"][1])
            expected = p["speed"] * (sim_time - p["chk_time"])
            if moved < STEP_MIN_RATIO * expected and sim_time - p["last_jump"] >= JUMP_COOLDOWN:
                p["actor"].set_location(carla.Location(
                    x=loc.x + dx / dist * STEP_UP_FORWARD,
                    y=loc.y + dy / dist * STEP_UP_FORWARD,
                    z=loc.z + STEP_UP_HEIGHT))
                p["last_jump"] = sim_time
                p["n_jumps"] += 1
                print(f"   ↑ 行人 {p['name']} 在 ({loc.x:.1f},{loc.y:.1f}) 前進受阻，跨上台階 (第 {p['n_jumps']} 次)")
            p["chk_pos"], p["chk_time"] = (loc.x, loc.y), sim_time

        p["actor"].apply_control(carla.WalkerControl(
            direction=carla.Vector3D(dx / dist, dy / dist, 0.0), speed=p["speed"], jump=False))

        # 卡住偵測：跳了仍長時間幾乎沒位移 -> 視為被擋住，結束此行人
        if p["last_pos"] is None or math.hypot(loc.x - p["last_pos"][0], loc.y - p["last_pos"][1]) > 0.3:
            p["last_pos"], p["still_since"] = (loc.x, loc.y), sim_time
        elif sim_time - p["still_since"] > STUCK_SECONDS:
            stop_walker(p)
            p["is_finished"] = True
            print(f"⚠️ 行人 {p['name']} 在 ({loc.x:.1f},{loc.y:.1f}) 卡住超過 {STUCK_SECONDS} 秒，"
                  "提前結束此行人 (請檢查該路線是否有障礙物)")

def save_resized(image, path):
    """把 CARLA 影像 (BGRA) 縮成 SAVE_SIZE x SAVE_SIZE 後存成 PNG。"""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
    if image.width != SAVE_SIZE:
        arr = cv2.resize(np.ascontiguousarray(arr), (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    global image_count, BASE_FRAME
    rel_frame = sensor_data.frame - BASE_FRAME          # ★ 相對 frame (每輪從 0 開始)
    if sensor_name == 'camera_Z':
        image_count += 1
        save_resized(sensor_data, os.path.join(output_path_image_Z, '%06d.png' % rel_frame))
    elif sensor_name == 'camera_Y':
        save_resized(sensor_data, os.path.join(output_path_image_Y, '%06d.png' % rel_frame))
    elif sensor_name == 'gnss_Z':
        gnss_data_Z.append([sensor_data.longitude, sensor_data.latitude])
    elif sensor_name == 'gnss_Y':
        gnss_data_Y.append([sensor_data.longitude, sensor_data.latitude])
    elif sensor_name == 'imu_Z':
        imu_data_Z.append([sensor_data.compass])
    elif sensor_name == 'imu_Y':
        imu_data_Y.append([sensor_data.compass])
    sensor_queue.put((rel_frame, sensor_name))


def parser():
    return ARGS   # 參數已在檔案開頭解析 (輸出資料夾 + --scenario + --sync)


def main():
    args = parser()
    faulthandler.dump_traceback_later(300, exit=True)   # 初始化 (載入地圖、生成車與行人) 最多 5 分鐘
    actors_list = []
    sensors_list = []
    active_pedestrians = []

    sensors_tick_time = str(0.5)
    FIXED_DELTA = 0.05
    MAX_SIM_SECONDS = 40

    world = None
    client = None
    origin_settings = None
    synchronous_master = False

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        # ★ 地圖已是 Town05_Opt 就沿用，不每輪重新 load_world (重複載入地圖是 CARLA 記憶體累積的主因之一)
        world = client.get_world()
        if not world.get_map().name.endswith(town):
            print(f"🗺️ 載入地圖 {town} ...")
            world = client.load_world(town)
        else:
            print(f"🗺️ 沿用已載入的地圖 {town}，清除上一輪可能殘留的實體...")
            s0 = world.get_settings()
            if s0.synchronous_mode:          # 上一輪若異常結束，世界可能還卡在同步模式
                s0.synchronous_mode = False
                s0.fixed_delta_seconds = None
                world.apply_settings(s0)
            actors = world.get_actors()
            ctrls = [a for a in actors if a.type_id.startswith("controller.ai.walker")]
            for c in ctrls:
                try:
                    c.stop()
                except Exception:
                    pass
            leftovers = (ctrls
                         + [a for a in actors if a.type_id.startswith("sensor.")]
                         + [a for a in actors if a.type_id.startswith("walker.")]
                         + [a for a in actors if a.type_id.startswith("vehicle.")])
            if leftovers:
                client.apply_batch_sync([carla.command.DestroyActor(a) for a in leftovers])
                print(f"   清除殘留實體 {len(leftovers)} 個")
            world.wait_for_tick()
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        carla_map = world.get_map()
        origin_settings = world.get_settings()
        blueprint_library = world.get_blueprint_library()
        world.set_weather(weather)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        if args.sync:
            settings = world.get_settings()
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = FIXED_DELTA
                world.apply_settings(settings)
            else:
                synchronous_master = True

        # === 兩台 (靜止) 車輛：位置由 SCENARIO 決定 ===
        poses = SCN
        zx, zy, zyaw = poses["Z"]
        yx, yy, yyaw = poses["Y"]
        print(f"🚗 場景 {SCENARIO}: Z=({zx},{zy},yaw={zyaw})  Y=({yx},{yy},yaw={yyaw})")
        tesla_blue = blueprint_library.find('vehicle.tesla.model3')
        tesla_blue.set_attribute('color', '0,0,255')
        tesla_red = blueprint_library.find('vehicle.tesla.model3')
        tesla_red.set_attribute('color', '255,0,0')
        vehicle_Z = world.spawn_actor(tesla_blue, carla.Transform(carla.Location(x=zx, y=zy, z=0.1), carla.Rotation(yaw=zyaw)))
        actors_list.append(vehicle_Z)
        vehicle_Y = world.spawn_actor(tesla_red, carla.Transform(carla.Location(x=yx, y=yy, z=0.1), carla.Rotation(yaw=yyaw)))
        actors_list.append(vehicle_Y)

        # 記錄場景標籤，之後合併資料時可分場景統計
        with open(os.path.join(output_path_ego, 'scenario.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{SCENARIO}\nZ={poses['Z']}\nY={poses['Y']}\nroute_seed={ROUTE_SEED}\n")

        # === 行人：固定路徑 (四個場景共用；同一 route-seed 軌跡完全相同) ===
        routes = load_route_set(ROUTE_SEED) or make_fixed_routes(ROUTE_SEED)
        active_pedestrians = spawn_fixed_pedestrians(world, blueprint_library, routes)
        with open(os.path.join(output_path_ego, 'pedestrian_routes.txt'), 'w', encoding='utf-8') as f:
            for r in routes:
                f.write(f"{r['id']} waypoints={r['waypoints']} speed={r['speed']} delay={r['delay']}\n")

        # === 感測器 (兩車相同 tick，維持同步) ===
        sensor_queue = Queue()
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute("sensor_tick", sensors_tick_time)
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute("sensor_tick", sensors_tick_time)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('sensor_tick', sensors_tick_time)
        camera_bp.set_attribute('image_size_x', '6144')   # (想加速可降到 1280x1280，保持 1:1)
        camera_bp.set_attribute('image_size_y', '6144')
        camera_bp.set_attribute('bloom_intensity', '0.0')
        camera_bp.set_attribute('lens_flare_intensity', '0.0')
        camera_bp.set_attribute('motion_blur_intensity', '0.0')
        camera_transform = carla.Transform(carla.Location(z=1.5))

        gnss_Z = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle_Z)
        gnss_Z.listen(lambda d: sensor_callback(d, sensor_queue, "gnss_Z")); sensors_list.append(gnss_Z)
        imu_Z = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle_Z)
        imu_Z.listen(lambda d: sensor_callback(d, sensor_queue, "imu_Z")); sensors_list.append(imu_Z)
        camera_Z = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle_Z)
        camera_Z.listen(lambda d: sensor_callback(d, sensor_queue, "camera_Z")); sensors_list.append(camera_Z)

        gnss_Y = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle_Y)
        gnss_Y.listen(lambda d: sensor_callback(d, sensor_queue, "gnss_Y")); sensors_list.append(gnss_Y)
        imu_Y = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle_Y)
        imu_Y.listen(lambda d: sensor_callback(d, sensor_queue, "imu_Y")); sensors_list.append(imu_Y)  # ★ 修正 imu_Y
        camera_Y = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle_Y)
        camera_Y.listen(lambda d: sensor_callback(d, sensor_queue, "camera_Y")); sensors_list.append(camera_Y)

        # ★ 設定本輪的 frame 起點：下一次 world.tick() 產生的 frame 編號就是 0
        global BASE_FRAME
        BASE_FRAME = world.get_snapshot().frame + 1
        print(f"⏱️ 本輪 frame 從 0 開始 (CARLA 原始 frame {BASE_FRAME} = 0)")
        with open(os.path.join(output_path_ego, 'scenario.txt'), 'a', encoding='utf-8') as f:
            f.write(f"base_frame={BASE_FRAME}\n")
        # === 主迴圈 ===
        max_ticks = int(MAX_SIM_SECONDS / FIXED_DELTA)
        tick_count = 0
        while True:
            spectator = world.get_spectator()
            # 俯視鏡頭放在兩車中點，兩車都看得到
            lz, ly = vehicle_Z.get_location(), vehicle_Y.get_location()
            mid = carla.Location(x=(lz.x + ly.x) / 2, y=(lz.y + ly.y) / 2, z=45)
            spectator.set_transform(carla.Transform(mid, carla.Rotation(pitch=-90)))

            if args.sync and synchronous_master:
                world.tick()
                faulthandler.dump_traceback_later(HANG_SECONDS, exit=True)   # ★ 每推進一步就重新計時
                tick_count += 1
                try:
                    for i in range(0, len(sensors_list)):
                        # ★ 第一筆只等 0.3 秒 (沒資料的步數很快跳過)；有資料時其餘幾筆最多等 2 秒
                        s_frame = sensor_queue.get(True, 0.3 if i == 0 else 2.0)
                        camera_frame = s_frame[0]
                        if s_frame[1] == "camera_Z":
                            geo_Z = carla_map.transform_to_geolocation(vehicle_Z.get_location())
                            geo_Y = carla_map.transform_to_geolocation(vehicle_Y.get_location())
                            frame_data = {"frame": camera_frame,
                                          "ego_Z_lon": geo_Z.longitude, "ego_Z_lat": geo_Z.latitude,
                                          "ego_Y_lon": geo_Y.longitude, "ego_Y_lat": geo_Y.latitude}
                            for ped_info in active_pedestrians:
                                p_name = ped_info["name"]
                                p_geo = carla_map.transform_to_geolocation(ped_info["actor"].get_location())
                                frame_data[f"{p_name}_lon"] = p_geo.longitude
                                frame_data[f"{p_name}_lat"] = p_geo.latitude
                                frame_data[f"{p_name}_dist_Z"] = haversine((geo_Z.latitude, geo_Z.longitude), (p_geo.latitude, p_geo.longitude), unit='m')
                                frame_data[f"{p_name}_dist_Y"] = haversine((geo_Y.latitude, geo_Y.longitude), (p_geo.latitude, p_geo.longitude), unit='m')
                            pedestrians_data_list.append(frame_data)
                            print(f"Frame {camera_frame}: 已同步紀錄 Z車 與 Y車 視角")
                except Empty:
                    print("Some of the sensor information is missed")
            else:
                world.wait_for_tick()

            # ★ 固定路徑行人控制 (錄影開始後的模擬時間)
            update_walkers(active_pedestrians, tick_count * FIXED_DELTA)

            if active_pedestrians and all(ped["is_finished"] for ped in active_pedestrians):
                print("\n🎉 所有行人皆已抵達目的地！準備結束並存檔...")
                break

            if tick_count >= max_ticks:
                print(f"\n⏱️ 已達最大模擬時間 {MAX_SIM_SECONDS} 秒，強制結束。")
                break

    finally:
        faulthandler.cancel_dump_traceback_later()
        print("\n=== 開始收尾與存檔 ===")
        try:
            print("...寫檔中...")
            if gnss_data_Z:
                pd.DataFrame(gnss_data_Z, columns=["vehicle_lon", "vehicle_lat"]).to_csv(os.path.join(output_path_ego, 'gps_Z.csv'), index=False)
            if gnss_data_Y:
                pd.DataFrame(gnss_data_Y, columns=["vehicle_lon", "vehicle_lat"]).to_csv(os.path.join(output_path_ego, 'gps_Y.csv'), index=False)
            if imu_data_Z:
                pd.DataFrame(imu_data_Z, columns=["orientation"]).to_csv(os.path.join(output_path_ego, 'imu_Z.csv'), index=False)
            if imu_data_Y:
                pd.DataFrame(imu_data_Y, columns=["orientation"]).to_csv(os.path.join(output_path_ego, 'imu_Y.csv'), index=False)
            if pedestrians_data_list:
                pd.DataFrame(pedestrians_data_list).to_csv(os.path.join(output_path_ego, 'pedestrians.csv'), index=False)
                print(f"✅ pedestrians.csv ({len(pedestrians_data_list)} 列)")
            else:
                print("⚠️ 沒有任何行人資料可寫入")
        except Exception as e:
            print(f"❌ 寫檔時發生錯誤: {e}")

        print('...開始銷毀 CARLA 實體...')
        try:
            if world is not None and origin_settings is not None:
                world.apply_settings(origin_settings)
            if client is not None:
                client.apply_batch([carla.command.DestroyActor(x) for x in actors_list])
                client.apply_batch([carla.command.DestroyActor(p["actor"]) for p in active_pedestrians if "actor" in p])
            for sensor in sensors_list:
                sensor.destroy()
            print("✅ 實體清理完畢")
        except Exception as e:
            print(f"⚠️ 清理實體時發生部分錯誤 (可忽略): {e}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')