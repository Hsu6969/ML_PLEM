import os
import sys
import pandas as pd
import numpy as np
import math

def vectorized_haversine(lat1, lon1, lat2, lon2):
    """使用 Numpy 進行極速 Haversine 距離計算 (回傳公尺)"""
    R = 6371000  # 地球半徑 (公尺)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def get_ped_location_input(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print(f"📂 開始提取 PLEM 預測特徵: {input_folder}")

    for filename in os.listdir(input_folder):
        # 配合我們上一步產生的檔名 (例如 four_slot_data_Z_final_P1.csv)
        if filename.startswith('four_slot_') and filename.endswith('.csv'):
            print(f"  ⏳ 正在轉換: {filename} ...", end="")
            df = pd.read_csv(os.path.join(input_folder, filename))
            
            if len(df) == 0:
                print(" 略過 (空檔案)")
                continue

            new_df = pd.DataFrame()
            new_df['frame'] = df['frame']
            
            # 1. 車速
            new_df['ego_speed'] = df['ego_speed']
            
            # 2~5. 角點座標 (替換成你最新的 tl_x, tl_y, br_x, br_y)
            new_df['x_leftup'] = df['tl_x']
            new_df['y_left_up'] = df['tl_y']
            new_df['x_rightbottom'] = df['br_x']
            new_df['y_rightbottom'] = df['br_y']
            
            # 6. BBOX_area (面積)
            new_df['BBOX_area'] = (df['br_x'] - df['tl_x']) * (df['br_y'] - df['tl_y'])
            
            # 7. BBOX_in_X (中心 X 座標 - 0.5)
            bbox_center_x = (df['tl_x'] + df['br_x']) / 2.0
            new_df['BBOX_in_X'] = bbox_center_x - 0.5
            
            # 8. BBox_in_time_X (橫向移動速度)
            new_df['BBox_in_time_X'] = (bbox_center_x - bbox_center_x.shift(1)).fillna(0)
            
            # 9. dist (使用極速向量化 Haversine)
            # 由於你的新資料中車輛座標叫 v_lat, v_lon，請確認這裡的名稱！
            v_lat_col = 'v_lat' if 'v_lat' in df.columns else 'ego_lat'
            v_lon_col = 'v_lon' if 'v_lon' in df.columns else 'ego_lon'
            
            dist_m = vectorized_haversine(df[v_lat_col], df[v_lon_col], df['p_lat'], df['p_lon'])
            new_df['dist'] = dist_m / 50.0  # 正規化
            
            # 10. gamma (使用 Numpy 向量化計算相對夾角)
            dx = df['p_lon'] - df[v_lon_col]
            dy = df['p_lat'] - df[v_lat_col]
            angle_rad = np.arctan2(dx, dy)
            angle_deg = np.degrees(angle_rad) % 360 # 確保在 0-360 之間
            beta_rad = np.radians(angle_deg)
            
            # 計算 gamma = 目標方位角 - 車輛朝向 (確保 orientation 也是弧度)
            gamma = beta_rad - df['orientation']
            # 將 gamma 包裝到 [-pi, pi] 的範圍內
            gamma = (gamma + np.pi) % (2 * np.pi) - np.pi
            # 依據原作者的邏輯進行正規化：(gamma + pi/4) / (pi/2)
            new_df['gamma'] = (gamma + (math.pi / 4)) / (math.pi / 2)

            # ==================================================
            # 歷史軌跡填補 (Dist & gamma)
            # ==================================================
            dist_first = new_df['dist'].iloc[0]
            gamma_first = new_df['gamma'].iloc[0]

            new_df['dist_1'] = new_df['dist'].shift(3).fillna(dist_first + 0.05)
            new_df['gamma_1'] = new_df['gamma'].shift(3).fillna(gamma_first + 0.003)
            
            new_df['dist_2'] = new_df['dist'].shift(2).fillna(dist_first + 0.03)
            new_df['gamma_2'] = new_df['gamma'].shift(2).fillna(gamma_first + 0.002)
            
            new_df['dist_3'] = new_df['dist'].shift(1).fillna(dist_first + 0.01)
            new_df['gamma_3'] = new_df['gamma'].shift(1).fillna(gamma_first + 0.001)

            # ==================================================
            # 反推 GPS 用的資訊保留
            # ==================================================
            new_df['ego_lat'] = df[v_lat_col]
            new_df['ego_lon'] = df[v_lon_col]
            new_df['target_lat'] = df['p_lat']
            new_df['target_lon'] = df['p_lon']
            new_df['ego_ori'] = df['orientation']
            new_df['inside/outside'] = df['inside/outside']
            new_df['p_NO'] = df['p_NO']

            # 輸出
            # 提取原始檔名中的 P1, P2 等 ID 資訊
            ped_part = filename.split('_')[-1].split('.')[0] # 取得 'P1' 等字串
            out_name = f'PLEM_input_{ped_part}.csv'
            new_df.to_csv(os.path.join(output_folder, out_name), index=False)
            print(f" ✅ 完成! (輸出: {out_name})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        # 手動測試時的路徑
        base_folder = r'D:\CARLA_Experiments\20260317_235036'

    # 讀取上一步驟產生出來的 four_slot 資料夾
    input_folder = os.path.join(base_folder, 'four_slot_original_data')
    # 建立最終給 PLEM 模型的資料夾
    output_folder = os.path.join(base_folder, 'PLEM_inputs')
    
    get_ped_location_input(input_folder, output_folder)
    print(f"\n🎉 恭喜！神經網路所需特徵已全數計算完畢，存於: {output_folder}")