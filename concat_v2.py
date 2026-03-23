import pandas as pd
import os
import sys

#將Precise_Vehicle_Placement_V5_2產生的各csv檔讀取並生成新的data

def merge_carla_data(folder_path):
    print(f"📂 正在處理資料夾: {folder_path}")
    
    # 1. 定義檔案路徑並讀取資料
    ped_path = os.path.join(folder_path, 'pedestrians.csv')
    imu_z_path = os.path.join(folder_path, 'imu_Z.csv')
    imu_y_path = os.path.join(folder_path, 'imu_Y.csv')

    try:
        df_ped = pd.read_csv(ped_path)
        df_imu_z = pd.read_csv(imu_z_path)
        df_imu_y = pd.read_csv(imu_y_path)
    except FileNotFoundError as e:
        print(f"❌ 找不到檔案: {e}")
        return

    # 2. 動態抓取行人編號 (自動尋找 P1_lon, P2_lon 等欄位)
    p_cols = [c for c in df_ped.columns if c.startswith('P') and '_lon' in c]
    p_nums = [c.split('_')[0].replace('P', '') for c in p_cols]

    # ==========================================
    # 處理 Z 車資料 (生成 data_Z.csv)
    # ==========================================
    # 建立 Z 車每一 Frame 的基礎特徵
    df_base_z = pd.DataFrame({
        "frame": df_ped["frame"],
        "ego_speed": 0,  # 依要求填入 0
        "v_lon": df_ped["ego_Z_lon"],
        "v_lat": df_ped["ego_Z_lat"],
        "orientation": df_imu_z["orientation"]
    })

    # 將所有行人的橫向資料展開成直向 (Long Format)
    ped_list_z = []
    for p_num in p_nums:
        temp_df = pd.DataFrame({
            "frame": df_ped["frame"],
            "p_lon": df_ped[f"P{p_num}_lon"],
            "p_lat": df_ped[f"P{p_num}_lat"],
            "p_NO": int(p_num)
        })
        ped_list_z.append(temp_df)

    # 合併行人與 Z 車基礎特徵
    df_peds_z = pd.concat(ped_list_z, ignore_index=True)
    df_final_z = pd.merge(df_base_z, df_peds_z, on="frame")
    
    # 依照 frame 和 p_NO 排序，並整理欄位順序以符合需求
    df_final_z = df_final_z.sort_values(by=["frame", "p_NO"]).reset_index(drop=True)
    df_final_z = df_final_z[["frame", "ego_speed", "v_lon", "v_lat", "orientation", "p_lon", "p_lat", "p_NO"]]
    df_final_z.to_csv(os.path.join(folder_path, 'data_Z.csv'), index=False)

    # ==========================================
    # 處理 Y 車資料 (生成 data_Y.csv)
    # ==========================================
    # 建立 Y 車每一 Frame 的基礎特徵
    df_base_y = pd.DataFrame({
        "frame": df_ped["frame"],
        "ego_speed": 0,  
        "v_lon": df_ped["ego_Y_lon"],
        "v_lat": df_ped["ego_Y_lat"],
        "orientation": df_imu_y["orientation"]
    })

    # 將所有行人的橫向資料展開成直向
    ped_list_y = []
    for p_num in p_nums:
        temp_df = pd.DataFrame({
            "frame": df_ped["frame"],
            "p_lon": df_ped[f"P{p_num}_lon"],
            "p_lat": df_ped[f"P{p_num}_lat"],
            "p_NO": int(p_num)
        })
        ped_list_y.append(temp_df)

    # 合併行人與 Y 車基礎特徵
    df_peds_y = pd.concat(ped_list_y, ignore_index=True)
    df_final_y = pd.merge(df_base_y, df_peds_y, on="frame")
    
    # 依照 frame 和 p_NO 排序，並整理欄位順序
    df_final_y = df_final_y.sort_values(by=["frame", "p_NO"]).reset_index(drop=True)
    df_final_y = df_final_y[["frame", "ego_speed", "v_lon", "v_lat", "orientation", "p_lon", "p_lat", "p_NO"]]
    df_final_y.to_csv(os.path.join(folder_path, 'data_Y.csv'), index=False)

    print("✅ 處理完成！已成功生成 data_Z.csv 與 data_Y.csv")

if __name__ == "__main__":
    # 💡 請將下方的路徑替換成你實際要處理的資料夾路徑 (例如截圖中的資料夾)
    # TARGET_FOLDER = r"D:\CARLA_Experiments\20260311_194509"
    # merge_carla_data(TARGET_FOLDER)

    if len(sys.argv) > 1:
        # 如果有 (代表是透過 auto_pipeline.py 執行的)，就接住那個路徑
        TARGET_FOLDER = sys.argv[1]
    else:
        # 如果沒有 (代表你自己手動按執行的)，就用一個預設路徑方便單獨測試
        TARGET_FOLDER = r"D:\CARLA_Experiments\default_test"
    
    merge_carla_data(TARGET_FOLDER)
