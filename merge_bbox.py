import pandas as pd
import os
import math

def calculate_expected_x(v_lon, v_lat, orientation_rad, p_lon, p_lat):
    """
    透過真實地理座標，精準計算行人應該出現在相機畫面中的哪一個水平位置 (0~1)
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
    expected_x = 0.5 + (right_dist / forward_dist) * 0.5
    return expected_x

def merge_yolo_to_long_csv_ultimate(base_folder, csv_path, yolo_labels_folder, output_filename="data_Z_final.csv"):
    print(f"📂 正在讀取並進行『由左至右強制對齊』: {csv_path}")

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
                    if parts[0] == '0': 
                        yolo_boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

        frame_indices = df[df['frame'] == frame_id].index
        if len(frame_indices) == 0: continue

        v_lon = df.at[frame_indices[0], 'v_lon']
        v_lat = df.at[frame_indices[0], 'v_lat']
        orientation = df.at[frame_indices[0], 'orientation']

        # ==========================================
        # ★ 步驟 1：計算理論位置，並將行人「由左至右」排序
        # ==========================================
        ped_expectations = []
        for idx in frame_indices:
            p_lon = df.at[idx, 'p_lon']
            p_lat = df.at[idx, 'p_lat']
            
            exp_x = calculate_expected_x(v_lon, v_lat, orientation, p_lon, p_lat)
            if exp_x is not None:
                ped_expectations.append({
                    'index': idx,
                    'expected_x': exp_x
                })
        
        # 將理論上的行人由左至右排序 (數值小到大)
        ped_expectations.sort(key=lambda p: p['expected_x'])

        # ==========================================
        # ★ 步驟 2：將 YOLO 抓到的框「由左至右」排序
        # ==========================================
        yolo_boxes.sort(key=lambda b: b[0])

        # ==========================================
        # ★ 步驟 3：智慧配對 (ZIP 對齊)
        # ==========================================
        if len(ped_expectations) == len(yolo_boxes):
            # 情況 A：偵測到的框數量剛好等於預期人數，直接完美對齊！
            for i in range(len(yolo_boxes)):
                idx = ped_expectations[i]['index']
                box = yolo_boxes[i]
                df.at[idx, 'client_x'] = box[0]
                df.at[idx, 'client_y'] = box[1]
                df.at[idx, 'width'] = box[2]
                df.at[idx, 'height'] = box[3]
                df.at[idx, 'inside/outside'] = 1
        else:
            # 情況 B：有人被遮蔽或沒偵測到，退回貪婪演算法 (但確保是由左至右分配)
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
        print(f"✅ 強制對齊配對完成！檔案已輸出至: {output_path}")
    except PermissionError:
        print(f"❌ 寫入失敗！請檢查 {output_filename} 是否正被 Excel 開啟，請關閉後再試一次！")

if __name__ == "__main__":
    BASE_DIR = r'D:\CARLA_Experiments\20260311_185919'
    CSV_Z_PATH = os.path.join(BASE_DIR, 'data_Z.csv')
    YOLO_Z_LABELS_DIR = os.path.join(BASE_DIR, r'image_Z\predict_result\predict\labels')
    CSV_Y_PATH = os.path.join(BASE_DIR, 'data_Y.csv')
    YOLO_Y_LABELS_DIR = os.path.join(BASE_DIR, r'image_Y\predict_result\predict\labels')
    
    merge_yolo_to_long_csv_ultimate(BASE_DIR, CSV_Z_PATH, YOLO_Z_LABELS_DIR, output_filename="data_Z_final.csv")
    merge_yolo_to_long_csv_ultimate(BASE_DIR, CSV_Y_PATH, YOLO_Y_LABELS_DIR, output_filename="data_Y_final.csv")