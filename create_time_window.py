import pandas as pd
import os
import sys

def create_four_slot_window(input_folder, output_folder):
    """
    使用向量化操作 (Vectorized) 建立 4-slot 滑動時間窗，速度極快
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"📂 開始處理資料夾: {input_folder}")

    for filename in os.listdir(input_folder):
        if filename.startswith('data_') and filename.endswith('.csv'):
            print(f"  ⏳ 正在轉換: {filename} ...", end="")
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # 如果資料太少無法組成連續特徵，就跳過
            if len(df) == 0:
                print(" 略過 (空檔案)")
                continue

            # ==========================================
            # ★ 定義需要「看過去歷史」的特徵欄位
            # (已經替換成你最新的 tl_x, br_y 格式)
            # ==========================================
            features_to_shift = [
                'v_lat', 'v_lon', 'p_lat', 'p_lon', 'orientation', 'ego_speed',
                'tl_x', 'tl_y', 'br_x', 'br_y', 'inside/outside'
            ]
            
            # 確保檔案裡真的有這些欄位
            existing_features = [f for f in features_to_shift if f in df.columns]

            new_df = pd.DataFrame()
            new_df['frame'] = df['frame']
            new_df['p_NO'] = df['p_NO']

            # ==========================================
            # ★ 核心魔法：使用 shift 快速取得過去資料
            # ==========================================
            for feat in existing_features:
                # 3幀前 (t-3) -> 對應你的 _1 或 _pass_1
                new_df[f'{feat}_1'] = df[feat].shift(3)
                # 2幀前 (t-2) -> 對應你的 _2 或 _pass_2
                new_df[f'{feat}_2'] = df[feat].shift(2)
                # 1幀前 (t-1) -> 對應你的 _3 或 _pass_3
                new_df[f'{feat}_3'] = df[feat].shift(1)
                # 現在 (t)
                new_df[feat] = df[feat]

            # ==========================================
            # ★ 處理開頭的空缺 (Padding)
            # 模仿你原本的邏輯：GPS和物理狀態用「複製第一筆」，BBox和畫面判定用「填0」
            # ==========================================
            gps_features = [col for col in new_df.columns if 'lat' in col or 'lon' in col or 'orientation' in col or 'speed' in col]
            bbox_features = [col for col in new_df.columns if '_x' in col or '_y' in col or 'inside/outside' in col]

            # 使用 bfill() (Backward Fill) 把第一筆資料往上填補
            new_df[gps_features] = new_df[gps_features].bfill()
            
            # BBox 相關的欄位開頭空缺補 0
            new_df[bbox_features] = new_df[bbox_features].fillna(0)

            # 儲存到新資料夾
            out_name = f"four_slot_{filename}"
            new_df.to_csv(os.path.join(output_folder, out_name), index=False)
            print(f" ✅ 完成! (輸出: {out_name})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        # 手動測試用
        base_folder = r'D:\CARLA_Experiments\20260317_235036'

    # 從上一個步驟的 'split_ped' 資料夾拿資料
    input_folder = os.path.join(base_folder, 'split_ped') 
    
    # 輸出到另一個專屬的 'four_slot_original_data' 資料夾
    output_folder = os.path.join(base_folder, 'four_slot_original_data')
    
    create_four_slot_window(input_folder, output_folder)
    print("\n🎉 所有時間窗 (Four Slot) 特徵轉換皆已完成！")