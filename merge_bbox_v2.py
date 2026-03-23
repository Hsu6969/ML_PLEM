import pandas as pd
import os
import math
import sys

def calculate_expected_x(v_lon, v_lat, orientation_rad, p_lon, p_lat):
    """
    透過真實地理座標，精準計算行人應該出現在相機畫面中的哪一個水平位置 (0~1)
    此部分與上一版相同
    """
    # 1. 將經緯度差轉換為公尺
    dx_east = (p_lon - v_lon) * 111320.0 * math.cos(math.radians(v_lat))
    dy_north = (p_lat - v_lat) * 111320.0 

    # 2. 找出車輛的「正前方」與「正右方」向量
    f_east = math.sin(orientation_rad)
    f_north = math.cos(orientation_rad)
    r_east = math.cos(orientation_rad)
    r_north = -math.sin(orientation_rad)

    # 3. 投影到車身座標系
    forward_dist = dx_east * f_east + dy_north * f_north
    right_dist = dx_east * r_east + dy_north * r_north

    # 若行人在車後，回傳 None
    if forward_dist <= 0:
        return None

    # 4. 算出理論畫面位置 (數值越大代表越靠右)
    # 此處做一個小小的改動：我們假設相機偏角較小，因此只用TAN值做投影
    # expected_x 的值範圍為 [0, 1]
    expected_x = 0.5 + 0.5 * (right_dist / forward_dist)
    return expected_x

def merge_yolo_to_long_csv_filtered(base_folder, csv_path, yolo_labels_folder, output_filename="data_Z_final.csv"):
    print(f"📂 正在讀取並進行『幾何投影配對 (含BBox大小過濾)』: {csv_path}")

    # ==========================================================
    # ★ 智慧幾何與大小過濾配對演算法
    # ==========================================================
    # 在此設定你認為合理的行人 bbox 大小範圍 (均為歸一化數值，0.0~1.0)
    # 你可以根據相機距離行人的遠近來調整這些值
    min_width = 0.015  # 行人 bbox 最小寬度
    max_width = 0.10  # 行人 bbox 最大寬度
    min_height = 0.08 # 行人 bbox 最小高度
    max_height = 0.20 # 行人 bbox 最大高度

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ 找不到 CSV 檔案: {csv_path}")
        return

    # 初始化空欄位
    df['client_x'] = ""
    df['client_y'] = ""
    df['width'] = ""
    df['height'] = ""
    df['inside/outside'] = 0

    unique_frames = df['frame'].unique()

    for frame_id in unique_frames:
        txt_filename = f"{int(frame_id):06d}.txt"
        txt_path = os.path.join(yolo_labels_folder, txt_filename)

        yolo_boxes = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts[0] == '0': # 確保只抓行人 (class 0)
                        
                        # 歸一化的 BBox 格式: [center_x, center_y, width, height]
                        x_c = float(parts[1])
                        y_c = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                        # ==================================================
                        # ★ 新增：大小過濾步驟
                        # ==================================================
                        # 檢查 BBox 的寬度和高度是否符合設定的範圍
                        # 這樣可以過濾掉太小或太大的誤判
                        if min_width <= w <= max_width and min_height <= h <= max_height:
                            # 只有符合範圍的 bbox 才會被加進去進行配對
                            yolo_boxes.append([x_c, y_c, w, h])
                        # 如果不符合範圍，我們就直接忽略這個偵測結果

        # ==========================================================
        # ★ 幾何投影配對邏輯 (與上一版相同，但操作的是過濾後的yolo_boxes)
        # ==========================================================
        frame_indices = df[df['frame'] == frame_id].index
        if len(frame_indices) == 0: continue

        v_lon = df.at[frame_indices[0], 'v_lon']
        v_lat = df.at[frame_indices[0], 'v_lat']
        orientation = df.at[frame_indices[0], 'orientation']

        # 計算理論位置，並將行人由左至右排序
        ped_expectations = []
        for idx in frame_indices:
            p_lon = df.at[idx, 'p_lon']
            p_lat = df.at[idx, 'p_lat']
            
            exp_x = calculate_expected_x(v_lon, v_lat, orientation, p_lon, p_lat)
            
            # 若行人在車前方 (exp_x 有值)，且理論上還在畫面上 (0 <= exp_x <= 1)
            # 為了防呆，我們稍微把範圍放寬一點到 [-0.1, 1.1] 
            if exp_x is not None and -0.1 <= exp_x <= 1.1:
                ped_expectations.append({
                    'index': idx,
                    'expected_x': exp_x
                })
        
        # 依照理論的 X 座標由小到大 (左到右) 進行排序
        ped_expectations.sort(key=lambda p: p['expected_x'])

        # 將過濾後的 YOLO 框也依照 X 座標由小到大排序
        yolo_boxes.sort(key=lambda b: b[0])

        # ==========================================
        # ★ ZIP拉鍊式對齊 (前提：偵測到的 valid BBox 數量剛好等於預期人數)
        # ==========================================
        if len(ped_expectations) == len(yolo_boxes):
            for i in range(len(yolo_boxes)):
                idx = ped_expectations[i]['index']
                box = yolo_boxes[i]
                
                # 配對成功！填入資料
                df.at[idx, 'client_x'] = box[0]
                df.at[idx, 'client_y'] = box[1]
                df.at[idx, 'width'] = box[2]
                df.at[idx, 'height'] = box[3]
                df.at[idx, 'inside/outside'] = 1
        
        else:
            # 情況 B：有人被遮蔽或大小不符沒偵測到，退回貪婪演算法 (但確保是由左至右分配)
            # 這雖然不是最完美的解法，但能防止某一幀整排空白。
            assigned_peds = set()
            for box in yolo_boxes:
                box_x = box[0]
                best_match_idx = -1
                min_dist = float('inf')

                for ped in ped_expectations:
                    if ped['index'] in assigned_peds:
                        continue
                    
                    dist = abs(ped['expected_x'] - box_x)
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = ped['index']

                # 如果距離在一個合理範圍內 (比如 0.3) 
                if best_match_idx != -1 and min_dist <= 0.3:
                    df.at[best_match_idx, 'client_x'] = box[0]
                    df.at[best_match_idx, 'client_y'] = box[1]
                    df.at[best_match_idx, 'width'] = box[2]
                    df.at[best_match_idx, 'height'] = box[3]
                    df.at[best_match_idx, 'inside/outside'] = 1
                    assigned_peds.add(best_match_idx)

    output_path = os.path.join(base_folder, output_filename)
    try:
        df.to_csv(output_path, index=False)
        print(f"✅ 強制對齊配對 (含大小過濾) 完成！檔案已輸出至: {output_path}")
    except PermissionError:
        print(f"❌ 寫入失敗！請檢查 {output_filename} 是否正被 Excel 開啟，請關閉後再試一次！")

# if __name__ == "__main__":
#     BASE_DIR = r'D:\CARLA_Experiments\20260311_185919'
#     CSV_Z_PATH = os.path.join(BASE_DIR, 'data_Z.csv')
#     YOLO_Z_LABELS_DIR = os.path.join(BASE_DIR, r'image_Z\predict_result\predict\labels')
#     CSV_Y_PATH = os.path.join(BASE_DIR, 'data_Y.csv')
#     YOLO_Y_LABELS_DIR = os.path.join(BASE_DIR, r'image_Y\predict_result\predict\labels')
    
#     merge_yolo_to_long_csv_filtered(BASE_DIR, CSV_Z_PATH, YOLO_Z_LABELS_DIR, output_filename="data_Z_final.csv")
#     merge_yolo_to_long_csv_filtered(BASE_DIR, CSV_Y_PATH, YOLO_Y_LABELS_DIR, output_filename="data_Y_final.csv")

if __name__ == "__main__":
    # ==============================================================
    # ★ 自動化管線接收參數修改
    # ==============================================================
    if len(sys.argv) > 1:
        BASE_DIR = sys.argv[1]
        print(f"🔗 [資料整併] 接收到指定資料夾路徑: {BASE_DIR}")
    else:
        # 手動測試時的預設路徑
        BASE_DIR = r'D:\CARLA_Experiments\default_test'
        print(f"⏰ [手動執行] 使用預設資料夾路徑: {BASE_DIR}")

    # --------------------------------------------------
    # 1. Z 車路徑設定與整併
    # --------------------------------------------------
    CSV_Z_PATH = os.path.join(BASE_DIR, 'data_Z.csv')
    YOLO_Z_LABELS_DIR = os.path.join(BASE_DIR, 'image_Z', 'predict_result', 'predict', 'labels')
    
    print("🔄 正在整併 Z 車資料...")
    merge_yolo_to_long_csv_filtered(BASE_DIR, CSV_Z_PATH, YOLO_Z_LABELS_DIR, output_filename="data_Z_final.csv")

    # --------------------------------------------------
    # 2. Y 車路徑設定與整併
    # --------------------------------------------------
    CSV_Y_PATH = os.path.join(BASE_DIR, 'data_Y.csv')
    YOLO_Y_LABELS_DIR = os.path.join(BASE_DIR, 'image_Y', 'predict_result', 'predict', 'labels')
    
    print("🔄 正在整併 Y 車資料...")
    merge_yolo_to_long_csv_filtered(BASE_DIR, CSV_Y_PATH, YOLO_Y_LABELS_DIR, output_filename="data_Y_final.csv")

    print("\n🎉 全部的資料處理與整併皆已完成！")