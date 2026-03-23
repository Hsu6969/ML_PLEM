import carla
import argparse
import os 
from queue import Queue
from queue import Empty
import random
random.seed(42)
import pandas as pd
import math
from haversine import haversine
import time
import os
from datetime import datetime  # ★ 新增：引入 datetime 模組來取得當下時間
import sys

# 儲存拍照時間的list
time_record = []
# 計算已生成照片張數
image_count=0
# 記錄感測器資料
gnss_total_data_list=[]
imu_total_data_list=[]

# 記錄所有行人的軌跡資料 (List of Dictionaries)
pedestrians_data_list = [] 

output_path_ego = None

# 偷天換日：如果最後一個參數不是以 '-' 開頭（代表它是我們傳進來的路徑）
# 我們就把他 pop() 取出來並從參數列中刪除，這樣 CARLA 就不會報錯了！
if len(sys.argv) > 1 and not sys.argv[-1].startswith('-'):
    output_path_ego = sys.argv.pop(-1)
    print(f"🔗 [自動化管線] 接收到指定資料夾路徑: {output_path_ego}")
else:
    # 保留手動執行的彈性
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = 'D:/CARLA_Experiments'
    output_path_ego = os.path.join(base_dir, current_time_str)
    print(f"⏰ [手動執行] 自行建立時間資料夾: {output_path_ego}")

print(f"📁 本次實驗數據將儲存於: {output_path_ego}")

"directory for outputs"
if not os.path.exists(output_path_ego):
    os.makedirs(output_path_ego)

# ★ 修改：建立 A 車與 B 車的獨立圖片資料夾
output_path_image_Z = os.path.join(output_path_ego, 'image_Z')
output_path_image_Y = os.path.join(output_path_ego, 'image_Y')
os.makedirs(output_path_image_Z, exist_ok=True)
os.makedirs(output_path_image_Y, exist_ok=True)

# ★ 修改：準備兩組獨立的 List 來存感測器資料
gnss_data_Z, gnss_data_Y = [], []
imu_data_Z, imu_data_Y = [], []
pedestrians_data_list = [] 
image_count = 0

town='Town05_Opt'

"天氣list"
weather_list=[
    carla.WeatherParameters.ClearNoon, #0
    carla.WeatherParameters.CloudyNoon, #1
    # ... (省略中間，保留你原本的設定)
    carla.WeatherParameters.SoftRainSunset, #13
]
weather=weather_list[0] 

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    global image_count
    
    # 處理相機
    if sensor_name == 'camera_Z':
        image_count += 1
        sensor_data.save_to_disk(os.path.join(output_path_image_Z, '%06d.png' % sensor_data.frame))
    elif sensor_name == 'camera_Y':
        sensor_data.save_to_disk(os.path.join(output_path_image_Y, '%06d.png' % sensor_data.frame))
     
    # 處理 GPS
    elif sensor_name == 'gnss_Z':
        gnss_data_Z.append([sensor_data.longitude, sensor_data.latitude])
    elif sensor_name == 'gnss_Y':
        gnss_data_Y.append([sensor_data.longitude, sensor_data.latitude])

    # 處理 IMU
    elif sensor_name == 'imu_Z':
        imu_data_Z.append([sensor_data.compass])
    elif sensor_name == 'imu_Y':
        imu_data_Y.append([sensor_data.compass])

    sensor_queue.put((sensor_data.frame, sensor_name))

def parser():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--sync',
        action='store_true',
        default=True,
        help='Synchronous mode execution')
    return argparser.parse_args()

def main():
    global start_time
    args = parser()
    start_time = time.time()
    
    actors_list = []
    sensors_list = []
    active_pedestrians = [] # 存放所有生成成功且需控制的行人資訊
    
    sensors_tick_time = str(0.5) # 每隔 0.5 秒 會拍一張照和記錄一筆數據 (1秒2偵)
    world = None 
    client = None
    origin_settings = None

    try:
        # === CARLA initialization ===
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        world = client.get_world()
        world = client.load_world(town)
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        map = world.get_map()
        origin_settings = world.get_settings()
        blueprint_library = world.get_blueprint_library()
        world.set_weather(weather)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_random_device_seed(42)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True) 
        traffic_manager.global_percentage_speed_difference(50)
        traffic_manager.set_respawn_dormant_vehicles(True)
        traffic_manager.set_boundaries_respawn_dormant_vehicles(100,200)

        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05 # 1 秒會產生20偵的畫面
                world.apply_settings(settings)

        # === Vehicle Spawning === 
        tesla_blue = blueprint_library.find('vehicle.tesla.model3')
        tesla_blue.set_attribute('color', '0,0,255')
        tesla_red = blueprint_library.find('vehicle.tesla.model3')
        tesla_red.set_attribute('color', '255,0,0')
        
        # X 車 
        spawn_point_Z = carla.Transform(carla.Location(x=-67.5, y=2.75, z=0.1), carla.Rotation(yaw=0))
        vehicle_Z = world.spawn_actor(tesla_blue, spawn_point_Z)
        actors_list.append(vehicle_Z)

        # Y 車 
        spawn_point_Y = carla.Transform(carla.Location(x=-67.5, y=6.1, z=0.1), carla.Rotation(yaw=0))
        vehicle_Y = world.spawn_actor(tesla_red, spawn_point_Y)
        actors_list.append(vehicle_Y)


        pedestrian_configs = [
            {
                "id": "P1",  
                "Ped_blueprint_ID": "walker.pedestrian.0001", #(米色衣女性)
                 "spawn_loc": carla.Location(x=-60.8, y=-8.0, z=1.0), 
                "destination": carla.Location(x=-60.8, y=15.0, z=1.0),
                "speed": 2.0  # 固定速度(由左往右)
            },
            {
                "id": "P2", 
                "Ped_blueprint_ID": "walker.pedestrian.0002",#(紫白衣男性)
                "spawn_loc": carla.Location(x=-60.5, y=11.0, z=1.0), 
                "destination": carla.Location(x=-60.5, y=-12.0, z=1.0), 
                "speed": 2.0  # 固定速度(由右往左)
            },
            {
                "id": "P3",  
                "Ped_blueprint_ID": "walker.pedestrian.0003",#(灰衣男性)
                "spawn_loc": carla.Location(x=-60.0, y=-8.0, z=1.0),
                "destination": carla.Location(x=-60.0, y=15.0, z=1.0),
                "speed": 2.0  # 固定速度(由左往右)
            }
        ]

        # 生成行人並綁定 AI 控制器
        print("正在生成行人與 AI 控制器...")
        walker_ai_bp = blueprint_library.find('controller.ai.walker')
        
        for config in pedestrian_configs:
            # ped_bp = random.choice(blueprint_library.filter("walker.pedestrian.*"))
            ped_bp = blueprint_library.find(config["Ped_blueprint_ID"])
            # 建議將行人設為無敵，避免在階梯邊緣發生詭異碰撞而死亡
            if ped_bp.has_attribute('is_invincible'):
                ped_bp.set_attribute('is_invincible', 'true')
                
            ped_spawn_point = carla.Transform(config["spawn_loc"])
            ped_actor = world.try_spawn_actor(ped_bp, ped_spawn_point)
            
            if ped_actor:
                # 生成 AI 控制器並附著在行人身上
                ai_controller = world.try_spawn_actor(walker_ai_bp, carla.Transform(), attach_to=ped_actor)
                
                if ai_controller:
                    active_pedestrians.append({
                        "name": config["id"],
                        "actor": ped_actor,
                        "ai_controller": ai_controller, # 記錄控制器
                        "destination": config["destination"],
                        "speed": config["speed"],
                        "is_finished": False  # ★ 新增：標記是否已經走到終點
                    })
                    print(f"成功生成行人: {config['id']}")

        # 重要：在啟動 AI 之前，必須先讓世界 tick 一次，讓實體確實存在於世界中
        world.tick()

        # 啟動並設定 AI 控制器
        for ped_info in active_pedestrians:
            ai = ped_info["ai_controller"]
            ai.start()
            ai.set_max_speed(ped_info["speed"])        # 告訴他走多快
            ai.go_to_location(ped_info["destination"]) # 告訴他要去哪裡


        # === Sensors Setup ===
        sensor_queue = Queue()

        # 1. 準備 GNSS 藍圖
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute("sensor_tick", sensors_tick_time)

        # 2. 準備 IMU 藍圖
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute("sensor_tick", sensors_tick_time)

        # 3. 準備 Camera 藍圖
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('sensor_tick', sensors_tick_time)
        camera_bp.set_attribute('image_size_x', '6144')
        camera_bp.set_attribute('image_size_y', '6144')
        camera_bp.set_attribute('bloom_intensity', '0.0')
        camera_bp.set_attribute('lens_flare_intensity', '0.0')
        camera_bp.set_attribute('motion_blur_intensity', '0.0')
        
        camera_transform = carla.Transform(carla.Location(z=1.5))
        # ==========================================


        # 【綁定 A 車的感測器】
        gnss_Z = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle_Z)
        gnss_Z.listen(lambda data: sensor_callback(data, sensor_queue, "gnss_Z"))
        sensors_list.append(gnss_Z)

        imu_Z = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle_Z)
        imu_Z.listen(lambda data: sensor_callback(data, sensor_queue, "imu_Z"))
        sensors_list.append(imu_Z)

        camera_Z = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle_Z)
        camera_Z.listen(lambda data: sensor_callback(data, sensor_queue, "camera_Z"))
        sensors_list.append(camera_Z)

        # 【綁定 B 車的感測器】
        gnss_Y = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle_Y)
        gnss_Y.listen(lambda data: sensor_callback(data, sensor_queue, "gnss_Y"))
        sensors_list.append(gnss_Y)

        imu_Y = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle_Y)
        imu_Y.listen(lambda data: sensor_callback(data, sensor_queue, "imu_Y"))
        sensors_list.append(imu_Z)

        camera_Y = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle_Y)
        camera_Y.listen(lambda data: sensor_callback(data, sensor_queue, "camera_Y"))
        sensors_list.append(camera_Y)


        while True:
            spectator = world.get_spectator()
            ego_transform = vehicle_Z.get_transform()
            spectator.set_transform(carla.Transform(ego_transform.location + carla.Location(z=30),
                                                    carla.Rotation(pitch=-90)))

            if args.sync and synchronous_master:
                world.tick()
                try:
                    for i in range(0, len(sensors_list)):
                        s_frame = sensor_queue.get(True, 1.0)
                        camera_frame = s_frame[0]   

                        # ★ 修改：我們只在收到 camera_Z 時做一次統整計算，避免重複紀錄
                        if s_frame[1] == "camera_Z":
                            geo_Z = map.transform_to_geolocation(vehicle_Z.get_location())
                            geo_Y = map.transform_to_geolocation(vehicle_Y.get_location())
                            
                            frame_data = {
                                "frame": camera_frame,
                                "ego_Z_lon": geo_Z.longitude,
                                "ego_Z_lat": geo_Z.latitude,
                                "ego_Y_lon": geo_Y.longitude,
                                "ego_Y_lat": geo_Y.latitude
                            }

                            for ped_info in active_pedestrians:
                                p_name = ped_info["name"]
                                p_actor = ped_info["actor"]
                                p_geo = map.transform_to_geolocation(p_actor.get_location())
                                
                                # 分別計算行人到 A 車與 B 車的距離
                                dist_to_Z = haversine((geo_Z.latitude, geo_Z.longitude), (p_geo.latitude, p_geo.longitude), unit='m')
                                dist_to_Y = haversine((geo_Y.latitude, geo_Y.longitude), (p_geo.latitude, p_geo.longitude), unit='m')
                                
                                frame_data[f"{p_name}_lon"] = p_geo.longitude
                                frame_data[f"{p_name}_lat"] = p_geo.latitude
                                frame_data[f"{p_name}_dist_Z"] = dist_to_Z
                                frame_data[f"{p_name}_dist_Y"] = dist_to_Y
                                
                            pedestrians_data_list.append(frame_data)
                            print(f"Frame {camera_frame}: 已同步紀錄 A車 與 B車 視角")

                except Empty:
                    print("Some of the sensor information is missed")
            else:
                world.wait_for_tick()

            # 檢查行人是否抵達目的地，到了就煞車
            for ped_info in active_pedestrians:
                # 如果他還沒走到終點，才需要檢查
                if not ped_info["is_finished"]:
                    current_loc = ped_info["actor"].get_location()
                    target_loc = ped_info["destination"]
                    
                    # 計算目前的 X, Y 與目標的 X, Y 的直線距離 (忽略 Z 軸高度差)
                    dist_to_target = math.sqrt((current_loc.x - target_loc.x)**2 + (current_loc.y - target_loc.y)**2)
                    
                    # 如果距離目標小於 1.5 公尺，視為已抵達
                    if dist_to_target < 1.5:
                        # 1. 停止 AI 控制器 (避免他亂走)
                        ped_info["ai_controller"].stop()
                        
                        # 2. 強制介入將速度設為 0，讓他原地站立
                        control = carla.WalkerControl(
                            direction=carla.Vector3D(0, 0, 0), 
                            speed=0.0, 
                            jump=False
                        )
                        ped_info["actor"].apply_control(control)
                        
                        # 3. 更新標籤，以後就不會再重複檢查他了
                        ped_info["is_finished"] = True
                        print(f"🚦 行人 {ped_info['name']} 已抵達目的地，停止移動。")

            # 檢查是否所有行人都已經走到目的地
            # all() 會檢查 active_pedestrians 裡面每一個 ped_info 的 "is_finished" 是否都變成 True 了
            if all(ped["is_finished"] for ped in active_pedestrians):
                print("\n🎉 所有行人皆已抵達目的地！準備結束模擬並儲存資料...")
                break  # 跳出 while True 無窮迴圈

    finally:
        print("\n=== 收到中斷指令，開始執行收尾與存檔 ===")
        
        # -------------------------------------------------
        # 1. 優先確保資料寫入 (放在最前面，避免被 CARLA 錯誤打斷)
        # -------------------------------------------------
        try:
            print("...寫檔中...")
           # 1. 寫入 A 車與 B 車的 GPS
            if gnss_data_Z:
                pd.DataFrame(gnss_data_Z, columns=["vehicle_lon", "vehicle_lat"]).to_csv(os.path.join(output_path_ego, 'gps_Z.csv'), index=False, mode="w")
            if gnss_data_Y:
                pd.DataFrame(gnss_data_Y, columns=["vehicle_lon", "vehicle_lat"]).to_csv(os.path.join(output_path_ego, 'gps_Y.csv'), index=False, mode="w")
            
            # 2. 寫入 A 車與 B 車的 IMU
            if imu_data_Z:
                pd.DataFrame(imu_data_Z, columns=["orientation"]).to_csv(os.path.join(output_path_ego, 'imu_Z.csv'), index=False, mode="w")
            if imu_data_Y:
                pd.DataFrame(imu_data_Y, columns=["orientation"]).to_csv(os.path.join(output_path_ego, 'imu_Y.csv'), index=False, mode="w")
            
            if pedestrians_data_list:
                all_ped_df = pd.DataFrame(pedestrians_data_list)
                all_ped_df.to_csv(os.path.join(output_path_ego, 'pedestrians.csv'), index=False, mode="w")
                print("✅ 已成功寫入多名行人軌跡至 pedestrians.csv")
            else:
                print("⚠️ 沒有任何 行人 資料可以寫入")
                
        except Exception as e:
            print(f"❌ 寫檔時發生錯誤: {e}")

        # -------------------------------------------------
        # 2. 清理 CARLA 實體 (包在 try-except 中，就算出錯也不影響已經存好的檔)
        # -------------------------------------------------
        print('...開始銷毀 CARLA 實體...')
        try:
            # ★ 修改：確認 world 真的有成功建立，才去還原設定
            if world is not None and origin_settings is not None:
                world.apply_settings(origin_settings)
            
            # ★ 修改：確認 client 真的有成功建立，才去砍掉物件
            if client is not None:
                client.apply_batch([carla.command.DestroyActor(x) for x in actors_list])
                # 確保刪除所有存活的行人
                client.apply_batch([carla.command.DestroyActor(p["actor"]) for p in active_pedestrians if "actor" in p])
                
            for sensor in sensors_list:
                sensor.destroy()
                
            for p in active_pedestrians:
                if "ai_controller" in p and p["ai_controller"] is not None:
                    p["ai_controller"].stop()
                    p["ai_controller"].destroy()
                if "actor" in p and p["actor"] is not None:
                    p["actor"].destroy()
            print("✅ 實體清理完畢")
            
        except Exception as e:
            print(f"⚠️ 清理實體時發生部分錯誤 (可忽略): {e}")
        
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')