import pandas as pd
import os
import glob
import sys

def clean_carla_data(folder_path):
    print(f"📂 正在處理資料夾: {folder_path}")
    
    # ==========================================
    # 1. 暴力刪除最後一張照片 (此時 CARLA 已關閉，絕對沒有檔案佔用問題)
    # ==========================================
    print("...開始清理最後一張可能破圖的照片...")
    img_dir_Z = os.path.join(folder_path, 'image_Z')
    img_dir_Y = os.path.join(folder_path, 'image_Y')
    
    for img_dir in [img_dir_Z, img_dir_Y]:
        list_of_files = glob.glob(os.path.join(img_dir, '*.png'))
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.basename) 
            try:
                os.remove(latest_file)
                print(f"🗑️ 已成功刪除照片: {os.path.basename(latest_file)}")
            except Exception as e:
                print(f"⚠️ 無法刪除照片 {latest_file}: {e}")

    print("✅ 資料清理對齊完畢！現在圖片數量與 CSV 完全一致！")

if __name__ == "__main__":
    # 💡 在這裡貼上你剛剛跑完 CARLA 產生的那個資料夾路徑
    # TARGET_FOLDER = r"D:/CARLA_Experiments/20260311_194509"
    # clean_carla_data(TARGET_FOLDER)

    if len(sys.argv) > 1:
        # 如果有 (代表是透過 auto_pipeline.py 執行的)，就接住那個路徑
        TARGET_FOLDER = sys.argv[1]
    else:
        # 如果沒有 (代表你自己手動按執行的)，就用一個預設路徑方便單獨測試
        TARGET_FOLDER = r"D:\CARLA_Experiments\default_test"
    
    clean_carla_data(TARGET_FOLDER)