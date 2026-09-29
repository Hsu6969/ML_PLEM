import pandas as pd
import os
import math
import sys

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==============================================================
# ★ v4 改動重點
#   1. 固定大小過濾 (h 0.08~0.20) -> 依距離自適應：用真值距離推估「應有高度」，
#      實際高度落在 0.5~2 倍內才算合理。車輛放哪都適用 (A+G 遠距行人不再被濾掉)。
#   2. ZIP 拉鍊 + 貪婪 -> 最佳指派 (匈牙利演算法)，成本 = 水平位置差 + 高度差。
#      兩人重疊只剩一個框時，框的大小會吻合「前面那個人」(近的人比較大)，
#      自動把框給前面的人，後面被遮住的人維持 inside=0 (本來就沒被看到，這是正確的)。
#   3. 門檻 (gating) 收緊：位置差 > 0.08 或高度比不合理的配對一律不收，避免 ID 錯配。
# ==============================================================

PED_HEIGHT_M = 1.8     # CARLA 行人約 1.7~1.9 m
FOCAL_NORM = 0.5       # FOV 90°、正方形影像：歸一化焦距 = 0.5
MAX_DX = 0.08          # 水平位置最大容許差 (歸一化)
HEIGHT_RATIO_RANGE = (0.5, 2.0)
W_HEIGHT = 0.5         # 成本中高度項權重


def calculate_expected(v_lon, v_lat, orientation_rad, p_lon, p_lat):
    """回傳 (expected_x, forward_dist)；行人在車後回傳 (None, None)"""
    dx_east = (p_lon - v_lon) * 111320.0 * math.cos(math.radians(v_lat))
    dy_north = (p_lat - v_lat) * 111320.0

    f_east, f_north = math.sin(orientation_rad), math.cos(orientation_rad)
    r_east, r_north = math.cos(orientation_rad), -math.sin(orientation_rad)

    forward_dist = dx_east * f_east + dy_north * f_north
    right_dist = dx_east * r_east + dy_north * r_north
    if forward_dist <= 0.5:
        return None, None
    expected_x = 0.5 + FOCAL_NORM * (right_dist / forward_dist)
    return expected_x, forward_dist


def match_cost(ped, box):
    """配對成本；不合理則回傳 None"""
    dx = abs(ped['expected_x'] - box[0])
    if dx > MAX_DX:
        return None
    exp_h = FOCAL_NORM * PED_HEIGHT_M / ped['forward']
    ratio = box[3] / exp_h
    if not (HEIGHT_RATIO_RANGE[0] <= ratio <= HEIGHT_RATIO_RANGE[1]):
        return None
    return dx + W_HEIGHT * abs(math.log(ratio))


def assign(peds, boxes):
    """回傳 [(ped_i, box_j), ...]"""
    if not peds or not boxes:
        return []
    BIG = 1e6
    cost = [[BIG] * len(boxes) for _ in peds]
    for i, p in enumerate(peds):
        for j, b in enumerate(boxes):
            c = match_cost(p, b)
            if c is not None:
                cost[i][j] = c

    pairs = []
    if HAS_SCIPY:
        rows, cols = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(rows, cols) if cost[i][j] < BIG]
    else:
        # 後備：依成本由小到大貪婪
        cand = sorted((cost[i][j], i, j) for i in range(len(peds))
                      for j in range(len(boxes)) if cost[i][j] < BIG)
        used_p, used_b = set(), set()
        for c, i, j in cand:
            if i not in used_p and j not in used_b:
                pairs.append((i, j)); used_p.add(i); used_b.add(j)
    return pairs


def merge_yolo_to_long_csv_filtered(base_folder, csv_path, yolo_labels_folder, output_filename="data_Z_final.csv"):
    print(f"📂 [v4] 幾何投影 + 距離自適應大小 + 最佳指派: {csv_path}")
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

    n_box_total, n_matched = 0, 0

    for frame_id in df['frame'].unique():
        txt_path = os.path.join(yolo_labels_folder, f"{int(frame_id):06d}.txt")
        boxes = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == '0':
                        x_c, y_c, w, h = map(float, parts[1:5])
                        # 只擋明顯離譜的框，其餘交給 match_cost 依距離判斷
                        if 0.003 <= h <= 0.6 and w <= 0.3:
                            boxes.append([x_c, y_c, w, h])
        n_box_total += len(boxes)

        frame_indices = df[df['frame'] == frame_id].index
        if len(frame_indices) == 0:
            continue
        v_lon = df.at[frame_indices[0], 'v_lon']
        v_lat = df.at[frame_indices[0], 'v_lat']
        orientation = df.at[frame_indices[0], 'orientation']

        peds = []
        for idx in frame_indices:
            exp_x, fwd = calculate_expected(v_lon, v_lat, orientation,
                                            df.at[idx, 'p_lon'], df.at[idx, 'p_lat'])
            if exp_x is not None and -0.1 <= exp_x <= 1.1:
                peds.append({'index': idx, 'expected_x': exp_x, 'forward': fwd})

        for i, j in assign(peds, boxes):
            idx, box = peds[i]['index'], boxes[j]
            df.at[idx, 'client_x'] = box[0]
            df.at[idx, 'client_y'] = box[1]
            df.at[idx, 'width'] = box[2]
            df.at[idx, 'height'] = box[3]
            df.at[idx, 'inside/outside'] = 1
            n_matched += 1

    print(f"   YOLO 框 {n_box_total} 個，成功配對 {n_matched} 個 (未配對 = 誤判/重複框/被遮擋)")
    output_path = os.path.join(base_folder, output_filename)
    try:
        df.to_csv(output_path, index=False)
        print(f"✅ 輸出: {output_path}")
    except PermissionError:
        print(f"❌ 寫入失敗！請確認 {output_filename} 沒有被 Excel 開啟")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        BASE_DIR = sys.argv[1]
        print(f"🔗 [資料整併] 接收到指定資料夾路徑: {BASE_DIR}")
    else:
        BASE_DIR = r'D:\CARLA_Experiments\default_test'
        print(f"⏰ [手動執行] 使用預設資料夾路徑: {BASE_DIR}")
    if not HAS_SCIPY:
        print("⚠️ 未安裝 scipy，改用貪婪指派 (建議 pip install scipy)")

    for car in ("Z", "Y"):
        print(f"🔄 正在整併 {car} 車資料...")
        merge_yolo_to_long_csv_filtered(
            BASE_DIR,
            os.path.join(BASE_DIR, f'data_{car}.csv'),
            os.path.join(BASE_DIR, f'image_{car}', 'predict_result', 'predict', 'labels'),
            output_filename=f"data_{car}_final.csv")

    print("\n🎉 全部的資料處理與整併皆已完成！")
