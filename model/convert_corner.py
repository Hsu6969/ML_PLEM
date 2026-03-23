import pandas as pd
import os
import sys

def convert_yolo_to_corners(csv_path):
    """
    將 CSV 中的 client_x, client_y, width, height 
    轉換成左上角 (tl_x, tl_y) 與右下角 (br_x, br_y)
    """
    print(f"📂 正在讀取並轉換 BBox 格式: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  ⚠️ 找不到檔案，跳過: {csv_path}")
        return

    # 確保原本的四個 YOLO 欄位存在
    yolo_cols = ['client_x', 'client_y', 'width', 'height']
    if not all(col in df.columns for col in yolo_cols):
        print(f"  ⚠️ 檔案缺少 YOLO BBox 欄位，跳過轉換。")
        return

    # 初始化新的座標欄位
    df['tl_x'] = ""
    df['tl_y'] = ""
    df['br_x'] = ""
    df['br_y'] = ""

    # 找出有 BBox 資料的列 (過濾掉空白或 NaN 的列)
    has_bbox = pd.notna(df['client_x']) & (df['client_x'] != "")

    if has_bbox.any():
        # 將字串轉為數值以便計算
        xc = pd.to_numeric(df.loc[has_bbox, 'client_x'])
        yc = pd.to_numeric(df.loc[has_bbox, 'client_y'])
        w = pd.to_numeric(df.loc[has_bbox, 'width'])
        h = pd.to_numeric(df.loc[has_bbox, 'height'])

        # 數學轉換邏輯：
        # 左上角 X = 中心 X - (寬度 / 2)
        # 左上角 Y = 中心 Y - (高度 / 2)
        # 右下角 X = 中心 X + (寬度 / 2)
        # 右下角 Y = 中心 Y + (高度 / 2)
        df.loc[has_bbox, 'tl_x'] = xc - (w / 2.0)
        df.loc[has_bbox, 'tl_y'] = yc - (h / 2.0)
        df.loc[has_bbox, 'br_x'] = xc + (w / 2.0)
        df.loc[has_bbox, 'br_y'] = yc + (h / 2.0)

    # 刪除舊的 YOLO 格式欄位
    df = df.drop(columns=yolo_cols)

    # 為了讓表格好看，調整欄位順序 (把 inside/outside 移到最後面)
    if 'inside/outside' in df.columns:
        cols = list(df.columns)
        cols.remove('inside/outside')
        cols.append('inside/outside')
        df = df[cols]

    # 直接覆蓋原檔案 (或你也可以存成新的檔名)
    try:
        df.to_csv(csv_path, index=False)
        print(f"  ✅ 轉換成功！已更新檔案。")
    except PermissionError:
        print(f"  ❌ 寫入失敗！請確認檔案沒有被 Excel 開啟。")

if __name__ == "__main__":
    # 支援自動化管線參數傳遞
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        # 手動測試時的路徑 (請換成你實際想測試的資料夾)
        base_folder = r'D:\CARLA_Experiments\20260317_235036'

    # 同時處理 Z 車與 Y 車的最終資料檔
    csv_z = os.path.join(base_folder, "data_Z_final.csv")
    csv_y = os.path.join(base_folder, "data_Y_final.csv")

    convert_yolo_to_corners(csv_z)
    convert_yolo_to_corners(csv_y)
    
    print("\n🎉 所有 BBox 座標轉換皆已完成！")