import pandas as pd
import os
import sys

def split_csv_by_pedestrian(csv_path, output_prefix, output_dir):
    """
    讀取 CSV 檔案，並根據 'p_NO' 欄位將資料拆分成多個獨立的 CSV 檔，
    並統一存放到指定的 output_dir 資料夾中。
    """
    print(f"📂 正在讀取並拆分: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  ⚠️ 找不到檔案，跳過: {csv_path}")
        return

    if 'p_NO' not in df.columns:
        print(f"  ⚠️ 檔案中找不到 'p_NO' 欄位，無法拆分。")
        return

    # 取得所有不重複的行人 ID (排除空值)
    ped_ids = df['p_NO'].dropna().unique()

    # 依序濾出每個行人的資料並存檔
    for p_id in ped_ids:
        # 過濾出該行人的所有 Frame
        ped_df = df[df['p_NO'] == p_id]
        
        # 轉換為整數，確保檔名乾淨 (例如 1.0 -> 1)
        p_id_int = int(p_id)
        
        # 建立新的檔名，例如 data_Z_final_P1.csv
        output_filename = f"{output_prefix}_P{p_id_int}.csv"
        
        # ★ 關鍵修改：將輸出路徑指向我們新建的 output_dir
        output_path = os.path.join(output_dir, output_filename)
        
        # 儲存
        try:
            ped_df.to_csv(output_path, index=False)
            print(f"  ✅ 成功匯出: {output_filename} (共 {len(ped_df)} 筆資料)")
        except PermissionError:
            print(f"  ❌ 寫入失敗: {output_filename}！請確認檔案沒有被 Excel 開啟。")

if __name__ == "__main__":
    # 支援自動化管線參數傳遞
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        # 手動測試時的路徑
        base_folder = r'D:\CARLA_Experiments\20260317_235036'

    csv_z = os.path.join(base_folder, "data_Z_final.csv")
    csv_y = os.path.join(base_folder, "data_Y_final.csv")

    # ==============================================================
    # ★ 新增：自動建立一個獨立的 'split_ped' 資料夾來收納切好的檔案
    # ==============================================================
    split_ped_dir = os.path.join(base_folder, 'split_ped')
    os.makedirs(split_ped_dir, exist_ok=True)
    print(f"\n📁 已建立/確認輸出資料夾: {split_ped_dir}")

    print("\n🔪 開始依據行人 ID 拆分資料...")
    
    # 拆分 Z 車與 Y 車資料，並傳入 split_ped_dir
    split_csv_by_pedestrian(csv_z, "data_Z_final", split_ped_dir)
    split_csv_by_pedestrian(csv_y, "data_Y_final", split_ped_dir)
    
    print(f"\n🎉 所有行人資料拆分皆已完成！檔案已統一存放於: {split_ped_dir}")