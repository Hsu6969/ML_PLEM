import pandas as pd
import os
import math
import itertools

def calculate_expected_x(v_lon, v_lat, orientation_rad, p_lon, p_lat, camera_offset=1.5):
    """
    透過真實地理座標，精準計算行人應該出現在相機畫面中的水平位置
    ★ 關鍵修正：加入 camera_offset 校正近距離視差 (Parallax)
    """
    # 1. 經緯度差轉換為公尺
    dx_east = (p_lon - v_lon) * 111320.0 * math.cos(math.radians(v_lat))
    dy_north = (p_lat - v_lat) * 111320.0 

    # 2. 車輛的前方與右方向量
    f_east = math.sin(orientation_rad)
    f_north = math.cos(orientation_rad)
    r_east = math.cos(orientation_rad)
    r_north = -math.sin(orientation_rad)

    # 3. 投影到相機座標系 (扣除相機在車身往前的偏移量，解決近距離交錯錯覺)
    forward_dist = (dx_east * f_east + dy_north * f_north) - camera_offset
    right_dist = dx_east * r_east + dy_north * r_north

    # 若行人在相機後方，回傳 None
    if forward_dist <= 0:
        return None

    # 4. 算出理論畫面位置 (數值越小越左邊，越大越右邊)
    expected_x = 0.5 + 0.5 * (right_dist / forward_dist)
    return expected_x

def merge_unified_pipeline(base_folder, csv_path, yolo_labels_folder, output_filename="data_Z_final.csv"):
    print(f"📂 正在執行『大統整配對 (過濾+視差校正+絕對順序)』: {csv_path}")

    # ==========================================
    # ★ 設定檔：BBox 大小過濾範圍
    # ==========================================
    min_width, max_width = 0.01, 0.20
    min_height, max_height = 0.05, 0.60

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ 找不到 CSV 檔案: {csv_path}")
        return

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
                        w, h = float(parts[3]), float(parts[4])
                        # 過濾異常大小的框 (剔除 YOLO 誤判)
                        if min_width <= w <= max_width and min_height <= h <= max_height:
                            yolo_boxes.append([float(parts[1]), float(parts[2]), w, h])

        frame_indices = df[df['frame'] == frame_id].index
        if len(frame_indices) == 0: continue

        v_lon = df.at[frame_indices[0], 'v_lon']
        v_lat = df.at[frame_indices[0], 'v_lat']
        orientation = df.at[frame_indices[0], 'orientation']

        # ==========================================
        # ★ 步驟 1：計算加入視差校正的理論位置，並由左至右排序
        # ==========================================
        ped_expectations = []
        for idx in frame_indices:
            p_lon = df.at[idx, 'p_lon']
            p_lat = df.at[idx, 'p_lat']
            
            # 使用 camera_offset=1.5 進行精準計算
            exp_x = calculate_expected_x(v_lon, v_lat, orientation, p_lon, p_lat, camera_offset=1.5)
            if exp_x is not None:
                ped_expectations.append({
                    'index': idx,
                    'expected_x': exp_x
                })
        
        # 理論行人由左至右排序
        ped_expectations.sort(key=lambda p: p['expected_x'])
        # 實際 YOLO 框由左至右排序
        yolo_boxes.sort(key=lambda b: b[0])

        n_peds = len(ped_expectations)
        n_boxes = len(yolo_boxes)
        best_match = []

        # ==========================================
        # ★ 步驟 2：不交叉的絕對順序配對
        # ==========================================
        if n_peds > 0 and n_boxes > 0:
            if n_peds == n_boxes:
                # 數量完美相等，直接一對一拉鍊式對齊
                best_match = list(zip(range(n_peds), range(n_boxes)))
            
            elif n_peds < n_boxes:
                # YOLO 還是有抓到多餘的誤判框 -> 挑出 n_peds 個框，嚴格保持左右順序
                best_cost = float('inf')
                for box_indices in itertools.combinations(range(n_boxes), n_peds):
                    cost = sum(abs(ped_expectations[i]['expected_x'] - yolo_boxes[j][0]) for i, j in enumerate(box_indices))
                    if cost < best_cost:
                        best_cost = cost
                        best_match = list(zip(range(n_peds), box_indices))
            
            else:
                # YOLO 漏抓了人 -> 挑出 n_boxes 個人來配對，嚴格保持左右順序
                best_cost = float('inf')
                for ped_indices in itertools.combinations(range(n_peds), n_boxes):
                    cost = sum(abs(ped_expectations[i]['expected_x'] - yolo_boxes[j][0]) for j, i in enumerate(ped_indices))
                    if cost < best_cost:
                        best_cost = cost
                        best_match = list(zip(ped_indices, range(n_boxes)))

        # ==========================================
        # ★ 步驟 3：寫入最終配對結果
        # ==========================================
        for p_idx, b_idx in best_match:
            idx = ped_expectations[p_idx]['index']
            box = yolo_boxes[b_idx]
            df.at[idx, 'client_x'] = box[0]
            df.at[idx, 'client_y'] = box[1]
            df.at[idx, 'width'] = box[2]
            df.at[idx, 'height'] = box[3]
            df.at[idx, 'inside/outside'] = 1

    output_path = os.path.join(base_folder, output_filename)
    try:
        df.to_csv(output_path, index=False)
        print(f"✅ 大統整配對完成！檔案已輸出至: {output_path}")
    except PermissionError:
        print(f"❌ 寫入失敗！請檢查 {output_filename} 是否正被 Excel 開啟，請關閉後再試一次！")


if __name__ == "__main__":
    BASE_DIR = r'D:\CARLA_Experiments\20260311_185919'
    CSV_Z_PATH = os.path.join(BASE_DIR, 'data_Z.csv')
    YOLO_Z_LABELS_DIR = os.path.join(BASE_DIR, r'image_Z\predict_result\predict\labels')
    CSV_Y_PATH = os.path.join(BASE_DIR, 'data_Y.csv')
    YOLO_Y_LABELS_DIR = os.path.join(BASE_DIR, r'image_Y\predict_result\predict\labels')
    
    merge_unified_pipeline(BASE_DIR, CSV_Z_PATH, YOLO_Z_LABELS_DIR, output_filename="data_Z_final.csv")
    merge_unified_pipeline(BASE_DIR, CSV_Y_PATH, YOLO_Y_LABELS_DIR, output_filename="data_Y_final.csv")