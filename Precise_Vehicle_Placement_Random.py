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

# ============================================================
# 全域資料容器
# ============================================================
image_count = 0
gnss_data_Z, gnss_data_Y = [], []
imu_data_Z, imu_data_Y = [], []
pedestrians_data_list = []

output_path_ego = None

if len(sys.argv) > 1 and not sys.argv[-1].startswith('-'):
    output_path_ego = sys.argv.pop(-1)
    print(f"🔗 [自動化管線] 接收到指定資料夾路徑: {output_path_ego}")
else:
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path_ego = os.path.join('D:/CARLA_Experiments', current_time_str)
    print(f"⏰ [手動執行] 自行建立時間資料夾: {output_path_ego}")

print(f"📁 本次實驗數據將儲存於: {output_path_ego}")

# ★ 每輪不同亂數種子 (用時間戳)：每輪配置不同，又能靠種子重現
_digits = "".join(re.findall(r"\d", os.path.basename(output_path_ego)))
_seed = int(_digits[-9:]) if _digits else int(time.time())
random.seed(_seed)
print(f"🎲 本輪亂數種子: {_seed}")

if not os.path.exists(output_path_ego):
    os.makedirs(output_path_ego)

output_path_image_Z = os.path.join(output_path_ego, 'image_Z')
output_path_image_Y = os.path.join(output_path_ego, 'image_Y')
os.makedirs(output_path_image_Z, exist_ok=True)
os.makedirs(output_path_image_Y, exist_ok=True)

town = 'Town05_Opt'
weather = carla.WeatherParameters.ClearNoon

PED_BLUEPRINTS = ["walker.pedestrian.0001", "walker.pedestrian.0002", "walker.pedestrian.0003"]

# ============================================================
# ★ 視野帶：只保留落在「兩車前方鏡頭範圍」的導航點
#   (兩車在 x≈-67.5、朝 +x；行人在其前方 x≈-64~-54 一帶穿越馬路)
#   低側 / 高側 中間留空 (車輛所在的 y≈2~5)，讓行人真的「穿越」過鏡頭
# ============================================================
VIEW_X_MIN, VIEW_X_MAX = -64.0, -54.0
LOW_Y_MIN, LOW_Y_MAX = -16.0, 1.0     # 馬路下側 (人行道/穿越起點)
HIGH_Y_MIN, HIGH_Y_MAX = 5.0, 18.0    # 馬路上側


def sample_nav_points(world, need_each, max_samples=6000):
    """反覆向 CARLA 行人導航網格要位置，只保留落在視野帶的點，
       分成低側/高側兩組回傳 (保證都在網格上、行人一定能走)。"""
    low, high = [], []
    for _ in range(max_samples):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if not (VIEW_X_MIN <= loc.x <= VIEW_X_MAX):
            continue
        if LOW_Y_MIN <= loc.y <= LOW_Y_MAX:
            low.append(loc)
        elif HIGH_Y_MIN <= loc.y <= HIGH_Y_MAX:
            high.append(loc)
        if len(low) >= need_each and len(high) >= need_each:
            break
    return low, high


def make_pedestrian_configs_nav(world, n=3):
    """用導航網格點，生成 n 位『穿越馬路(上下向)』的行人配置。
       起點/終點分屬低側與高側 -> 穿越；方向隨機 -> 不同人常走相反向。
       導航點不足時回傳 None，交由外層整輪重試。"""
    low, high = sample_nav_points(world, need_each=n + 2)
    if len(low) < n or len(high) < n:
        print(f"   ⚠️ 視野帶內導航點不足 (低側 {len(low)} / 高側 {len(high)}，需各 {n})")
        return None

    random.shuffle(low)
    random.shuffle(high)
    configs = []
    for i in range(n):
        if random.random() < 0.5:
            start, dest = low[i], high[i]        # 由下往上穿越
        else:
            start, dest = high[i], low[i]        # 由上往下穿越
        configs.append({
            "id": f"P{i + 1}",
            "Ped_blueprint_ID": PED_BLUEPRINTS[i],
            "spawn_loc": carla.Location(x=start.x, y=start.y, z=1.0),
            "destination": carla.Location(x=dest.x, y=dest.y, z=dest.z),
            "speed": round(random.uniform(1.0, 2.0), 1),
        })
    print("🚶 本輪行人配置 (導航網格點):")
    for c in configs:
        print(f"   {c['id']}: ({c['spawn_loc'].x:.1f},{c['spawn_loc'].y:.1f}) → "
              f"({c['destination'].x:.1f},{c['destination'].y:.1f}), speed={c['speed']}")
    return configs


def _destroy_peds(active):
    for p in active:
        try:
            if p.get("ai_controller") is not None:
                p["ai_controller"].stop()
                p["ai_controller"].destroy()
            if p.get("actor") is not None:
                p["actor"].destroy()
        except Exception:
            pass


def spawn_and_validate_pedestrians(world, blueprint_library, configs,
                                   warmup_ticks=20, move_thresh=0.3):
    """生成行人 + AI，暖身數個 tick 確認『每個行人真的有在動』。
       任一個沒動或生成失敗 -> 清掉全部並回傳 None (整輪重生)。"""
    walker_ai_bp = blueprint_library.find('controller.ai.walker')
    active = []

    for config in configs:
        ped_bp = blueprint_library.find(config["Ped_blueprint_ID"])
        if ped_bp.has_attribute('is_invincible'):
            ped_bp.set_attribute('is_invincible', 'true')
        ped_actor = world.try_spawn_actor(ped_bp, carla.Transform(config["spawn_loc"]))
        if ped_actor is None:
            print(f"   ✗ {config['id']} 生成失敗")
            _destroy_peds(active)
            return None
        ai = world.try_spawn_actor(walker_ai_bp, carla.Transform(), attach_to=ped_actor)
        if ai is None:
            print(f"   ✗ {config['id']} AI 控制器生成失敗")
            ped_actor.destroy()
            _destroy_peds(active)
            return None
        active.append({"name": config["id"], "actor": ped_actor, "ai_controller": ai,
                       "destination": config["destination"], "speed": config["speed"],
                       "is_finished": False})

    world.tick()  # 讓實體確實存在
    start_locs = {p["name"]: p["actor"].get_location() for p in active}

    for p in active:
        p["ai_controller"].start()
        p["ai_controller"].set_max_speed(p["speed"])
        p["ai_controller"].go_to_location(p["destination"])

    # 暖身：確認每個行人真的有位移 (沒動的話代表卡住/走不了)
    for _ in range(warmup_ticks):
        world.tick()

    for p in active:
        now = p["actor"].get_location()
        s = start_locs[p["name"]]
        moved = math.sqrt((now.x - s.x) ** 2 + (now.y - s.y) ** 2)
        if moved < move_thresh:
            print(f"   ✗ {p['name']} 暖身 {warmup_ticks} tick 幾乎沒動 (moved={moved:.2f}m) -> 整輪重生")
            _destroy_peds(active)
            return None

    print("   ✓ 三位行人都確認會走")
    return active


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    global image_count
    if sensor_name == 'camera_Z':
        image_count += 1
        sensor_data.save_to_disk(os.path.join(output_path_image_Z, '%06d.png' % sensor_data.frame))
    elif sensor_name == 'camera_Y':
        sensor_data.save_to_disk(os.path.join(output_path_image_Y, '%06d.png' % sensor_data.frame))
    elif sensor_name == 'gnss_Z':
        gnss_data_Z.append([sensor_data.longitude, sensor_data.latitude])
    elif sensor_name == 'gnss_Y':
        gnss_data_Y.append([sensor_data.longitude, sensor_data.latitude])
    elif sensor_name == 'imu_Z':
        imu_data_Z.append([sensor_data.compass])
    elif sensor_name == 'imu_Y':
        imu_data_Y.append([sensor_data.compass])
    sensor_queue.put((sensor_data.frame, sensor_name))


def parser():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--sync', action='store_true', default=True, help='Synchronous mode')
    return argparser.parse_args()


def main():
    args = parser()
    actors_list = []
    sensors_list = []
    active_pedestrians = []

    sensors_tick_time = str(0.5)
    FIXED_DELTA = 0.05
    MAX_SIM_SECONDS = 180
    MAX_ATTEMPTS = 15               # 整輪重生的最大嘗試次數

    world = None
    client = None
    origin_settings = None
    synchronous_master = False

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        world = client.load_world(town)
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

        # === 兩台 (靜止) 車輛 ===
        tesla_blue = blueprint_library.find('vehicle.tesla.model3')
        tesla_blue.set_attribute('color', '0,0,255')
        tesla_red = blueprint_library.find('vehicle.tesla.model3')
        tesla_red.set_attribute('color', '255,0,0')
        vehicle_Z = world.spawn_actor(tesla_blue, carla.Transform(carla.Location(x=-67.5, y=2.75, z=0.1), carla.Rotation(yaw=0)))
        actors_list.append(vehicle_Z)
        vehicle_Y = world.spawn_actor(tesla_red, carla.Transform(carla.Location(x=-67.5, y=6.1, z=0.1), carla.Rotation(yaw=0)))
        actors_list.append(vehicle_Y)

        # === 行人：導航網格取點 + 暖身驗證 + 整輪重生 ===
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"\n🎲 產生行人配置 (第 {attempt}/{MAX_ATTEMPTS} 次嘗試)...")
            configs = make_pedestrian_configs_nav(world, n=3)
            if configs is None:
                continue
            active_pedestrians = spawn_and_validate_pedestrians(world, blueprint_library, configs)
            if active_pedestrians:
                break
            active_pedestrians = []

        if not active_pedestrians:
            raise RuntimeError("多次嘗試後仍無法生成可正常行走的行人；"
                               "請確認 VIEW_X / LOW_Y / HIGH_Y 範圍有落在行人導航網格上。")

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

        # === 主迴圈 ===
        max_ticks = int(MAX_SIM_SECONDS / FIXED_DELTA)
        tick_count = 0
        while True:
            spectator = world.get_spectator()
            ego_transform = vehicle_Z.get_transform()
            spectator.set_transform(carla.Transform(ego_transform.location + carla.Location(z=30),
                                                    carla.Rotation(pitch=-90)))

            if args.sync and synchronous_master:
                world.tick()
                tick_count += 1
                try:
                    for i in range(0, len(sensors_list)):
                        s_frame = sensor_queue.get(True, 1.0)
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

            for ped_info in active_pedestrians:
                if not ped_info["is_finished"]:
                    cur = ped_info["actor"].get_location()
                    tgt = ped_info["destination"]
                    if math.sqrt((cur.x - tgt.x) ** 2 + (cur.y - tgt.y) ** 2) < 1.5:
                        ped_info["ai_controller"].stop()
                        ped_info["actor"].apply_control(carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0, jump=False))
                        ped_info["is_finished"] = True
                        print(f"🚦 行人 {ped_info['name']} 已抵達目的地，停止移動。")

            if active_pedestrians and all(ped["is_finished"] for ped in active_pedestrians):
                print("\n🎉 所有行人皆已抵達目的地！準備結束並存檔...")
                break

            if tick_count >= max_ticks:
                print(f"\n⏱️ 已達最大模擬時間 {MAX_SIM_SECONDS} 秒，強制結束。")
                break

    finally:
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
            for p in active_pedestrians:
                ai = p.get("ai_controller")
                if ai is not None:
                    ai.stop(); ai.destroy()
            print("✅ 實體清理完畢")
        except Exception as e:
            print(f"⚠️ 清理實體時發生部分錯誤 (可忽略): {e}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')