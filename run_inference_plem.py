import os
import sys
import time
import math
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# 匯入你原本的自定義網路結構
import model.loss_function_PLEM as loss_function_PLEM
import model.layer_PLEM as layer_PLEM

def vectorized_reverse_gps(ego_lat, ego_lon, ego_ori, pred_dist_norm, pred_gamma_norm):
    """使用 Numpy 極速反推預測的 GPS 經緯度"""
    R = 6371000.0  # 地球半徑 (公尺)
    
    # 1. 還原真實距離 (m) 與 夾角 (rad)
    dist_m = pred_dist_norm * 50.0
    gamma_rad = (pred_gamma_norm * (math.pi / 2.0)) - (math.pi / 4.0)
    
    # 2. 計算目標絕對方位角 (bearing)
    bearing_rad = ego_ori + gamma_rad
    
    # 3. 經緯度轉弧度
    lat1 = np.radians(ego_lat)
    lon1 = np.radians(ego_lon)
    
    # 4. 向量化極速推算目標經緯度
    lat2 = np.arcsin(np.sin(lat1) * np.cos(dist_m / R) +
                     np.cos(lat1) * np.sin(dist_m / R) * np.cos(bearing_rad))
    lon2 = lon1 + np.arctan2(np.sin(bearing_rad) * np.sin(dist_m / R) * np.cos(lat1),
                             np.cos(dist_m / R) - np.sin(lat1) * np.sin(lat2))
    
    return np.degrees(lat2), np.degrees(lon2)

def vectorized_haversine_error(lat1, lon1, lat2, lon2):
    """使用 Numpy 極速計算兩點 GPS 的實際誤差距離 (公尺)"""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def run_plem_inference(input_folder, output_folder, base_dir):
    os.makedirs(output_folder, exist_ok=True)
    
    # ==========================================
    # 1. 載入模型與正規化工具 (Scaler)
    # ==========================================
    model_file_name = "model_20240716154220.h5"
    feedback_PLEM_model_path = r"D:/machine_learning/" + model_file_name
    speed_scaler_path = "D://machine_learning/0716_speed_minMaxScaler.save"
    area_scaler_path = "D://machine_learning/0716_area_minMaxScaler.save"

    print("🧠 正在載入 Feedback PLEM 模型與 Scaler...")
    
    # 1. 註冊 Custom Objects (左邊字串必須是舊的 PLENN，右邊對應到你現在新的 PLEM Class)
    tf.keras.utils.get_custom_objects()["Feedbacklayer_PLENN"] = layer_PLEM.Feedbacklayer_PLEM
    tf.keras.utils.get_custom_objects()['loss_for_feedback_PLENN'] = loss_function_PLEM.loss_for_feedback_PLEM
    
    try:
        # 2. 讀取模型 (同樣的，字典的 key 必須順從 .h5 的記憶，寫舊的 PLENN)
        model = tf.keras.models.load_model(
            feedback_PLEM_model_path, 
            custom_objects={
                'Feedbacklayer_PLENN': layer_PLEM.Feedbacklayer_PLEM,
                'loss_for_feedback_PLENN': loss_function_PLEM.loss_for_feedback_PLEM
            }
        )
        speed_scaler = joblib.load(speed_scaler_path)
        area_scaler = joblib.load(area_scaler_path)
    except Exception as e:
        print(f"❌ 模型或 Scaler 載入失敗，請確認路徑: {e}")
        return

    # ==========================================
    # 2. 開始批次處理資料
    # ==========================================
    for filename in os.listdir(input_folder):
        if filename.startswith('PLEM_input_') and filename.endswith('.csv'):
            print(f"  ⏳ 正在推論: {filename} ...", end="")
            
            # 提取行人 ID
            ped_id = filename.split('_')[-1].split('.')[0]
            data_ped = pd.read_csv(os.path.join(input_folder, filename))
            
            if len(data_ped) == 0:
                print(" 略過 (空檔案)")
                continue

            # ★ 安全的欄位名稱抓取法 (告別危險的 iloc 數字)
            now_cols = ['ego_speed', 'x_leftup', 'y_left_up', 'x_rightbottom', 'y_rightbottom', 'BBOX_area', 'BBOX_in_X', 'BBox_in_time_X']
            past_cols = ['dist_1', 'gamma_1', 'dist_2', 'gamma_2', 'dist_3', 'gamma_3']
            
            now_data = data_ped[now_cols].copy()
            past_data = data_ped[past_cols].copy()

            # ★ 進行特徵正規化
            # 將 dataframe 轉換為 numpy 陣列進行 transform 以免報錯
            now_data['ego_speed'] = speed_scaler.transform(now_data[['ego_speed']].values)
            now_data['BBOX_area'] = area_scaler.transform(now_data[['BBOX_area']].values)
            now_data['BBOX_in_X'] = now_data['BBOX_in_X'] / 0.5
            now_data['BBox_in_time_X'] = now_data['BBox_in_time_X'] / 0.5

            # ==========================================
            # 3. 模型預測
            # ==========================================
            # 餵給模型，取得預測結果 (轉換維度確保可以直接塞入 dataframe)
            preds = model.predict([now_data.values, past_data.values], verbose=0)
            data_ped['predict_dist'] = preds[:, 0]
            data_ped['predict_gamma'] = preds[:, 1]

            # ==========================================
            # 4. 極速還原 GPS 與計算誤差 (Numpy 向量化)
            # ==========================================
            pred_lat, pred_lon = vectorized_reverse_gps(
                data_ped['ego_lat'].values, 
                data_ped['ego_lon'].values, 
                data_ped['ego_ori'].values, 
                data_ped['predict_dist'].values, 
                data_ped['predict_gamma'].values
            )
            data_ped['predict_lat'] = pred_lat
            data_ped['predict_lon'] = pred_lon

            # 計算預測 GPS 與真實 GPS 的公尺誤差
            data_ped['GPS_difference'] = vectorized_haversine_error(
                data_ped['target_lat'].values, 
                data_ped['target_lon'].values, 
                data_ped['predict_lat'].values, 
                data_ped['predict_lon'].values
            )

            # 輸出檔案
            out_name = f'inference_GPS_{ped_id}.csv'
            data_ped.to_csv(os.path.join(output_folder, out_name), index=False)
            print(f" ✅ 完成! (輸出: {out_name})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        # 手動測試用路徑
        base_folder = r'D:\CARLA_Experiments\20260317_235036'

    # 輸入：上一步驟產生的 PLEM_inputs 資料夾
    input_folder = os.path.join(base_folder, 'PLEM_inputs')
    # 輸出：最終的預測結果與誤差
    output_folder = os.path.join(base_folder, 'inference_feedback_PLEM')

    run_plem_inference(input_folder, output_folder, base_folder)
    print("\n🎉 完美收工！所有行人的 GPS 位置預測及誤差計算皆已完成。")