import subprocess
import os
import sys
import time

def run_script_in_env(env_name, script_name, cwd, target_folder):
    """
    在指定的 Conda 環境中執行 Python 腳本，並將目標資料夾路徑動態傳遞給該程式
    """
    print(f"\n{'-'*65}")
    print(f"🚀 啟動: {script_name} (環境: {env_name})")
    
    cmd = ["conda", "run", "--no-capture-output", "-n", env_name, "python", script_name, target_folder]
    result = subprocess.run(cmd, cwd=cwd)
    
    if result.returncode != 0:
        print(f"\n❌ [錯誤] {script_name} 執行失敗！自動管線已安全中斷，保護資料不被污染。")
        sys.exit(1)

def main():
    # 統一設定工作目錄與虛擬環境名稱
    WORK_DIR = r"D:\machine_learning"
    ENV_CARLA = "carla37"
    ENV_YOLO = "yolov8_cu12"

    # ==================================================
    # ★ 設定你要循環自動跑幾次實驗 (例如: 1次)
    # ==================================================
    TOTAL_LOOPS = 1

    print(f"🎬 [全自動化管線啟動] 準備執行 {TOTAL_LOOPS} 次完整的資料生成與推論任務...")

    for i in range(TOTAL_LOOPS):
        print(f"\n{'='*70}")
        print(f"🔄 開始執行第 {i+1} / {TOTAL_LOOPS} 次實驗循環")
        print(f"{'='*70}")

        # ★ 動態建立專屬時間戳記資料夾，確保每次實驗資料獨立不覆蓋
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        EXPERIMENT_FOLDER = rf"D:\CARLA_Experiments\{timestamp}"
        print(f"📁 本次循環的目標資料夾: {EXPERIMENT_FOLDER}")

        # --------------------------------------------------
        # 階段一：CARLA 數據收集與清理 (環境: carla37)
        # --------------------------------------------------
        # 1. 開啟 CARLA 模擬器，控制車輛與行人，收集 GPS/IMU 並拍下畫面
        run_script_in_env(ENV_CARLA, "Precise_Vehicle_Placement_V5_2.py", WORK_DIR, EXPERIMENT_FOLDER)
        
        # 2. 清理無效或殘缺的初始數據
        run_script_in_env(ENV_CARLA, "clean_data.py", WORK_DIR, EXPERIMENT_FOLDER)
        
        # 3. 將清理後的零碎資料進行初步串接成 data_Z.csv / data_Y.csv
        run_script_in_env(ENV_CARLA, "concat_v2.py", WORK_DIR, EXPERIMENT_FOLDER)

        # --------------------------------------------------
        # 階段二：YOLO 視覺偵測與追蹤對齊 (環境: yolov8_cu12)
        # --------------------------------------------------
        # 4. 讀取 CARLA 拍下來的照片，透過 YOLO 找出畫面中的行人 BBox (產出 TXT)
        run_script_in_env(ENV_YOLO, "predict.py", WORK_DIR, EXPERIMENT_FOLDER)
        
        # 5. 結合視差校正與數學排序，將 YOLO 的 BBox 完美配對給正確的行人 ID (解決 ID Switch)
        run_script_in_env(ENV_YOLO, "merge_bbox_v2.py", WORK_DIR, EXPERIMENT_FOLDER)

        # --------------------------------------------------
        # 階段三：資料後處理與幾何轉換 (環境: yolov8_cu12)
        # --------------------------------------------------
        # 6. 將 YOLO 的 [中心點+長寬] 轉換為主流的 [左上角+右下角] 絕對座標
        run_script_in_env(ENV_YOLO, "convert_corners.py", WORK_DIR, EXPERIMENT_FOLDER)
        
        # 7. 從大表中把不同行人的資料獨立切分開來 (P1, P2, P3...)，存入 split_ped 資料夾
        run_script_in_env(ENV_YOLO, "split_pedestrians.py", WORK_DIR, EXPERIMENT_FOLDER)

        # --------------------------------------------------
        # 階段四：時間序列特徵工程 (環境: yolov8_cu12)
        # --------------------------------------------------
        # 8. 製作連續 4 幀的滑動時間窗 (Sliding Window)，讓模型能看見歷史軌跡
        run_script_in_env(ENV_YOLO, "create_time_window.py", WORK_DIR, EXPERIMENT_FOLDER)
        
        # 9. 透過 Numpy 極速向量化，計算真實距離 (dist) 與 相對方位角 (gemma) 並進行特徵正規化
        run_script_in_env(ENV_YOLO, "get_plem_features.py", WORK_DIR, EXPERIMENT_FOLDER)

        # --------------------------------------------------
        # 階段五：神經網路 (PLEM) 預測與驗證 (環境: yolov8_cu12)
        # --------------------------------------------------
        # 10. 載入 2024 年訓練的 H5 模型，餵入特徵進行預測，並反推回地球真實 GPS 座標，計算最終公尺誤差
        run_script_in_env(ENV_YOLO, "run_inference_plem.py", WORK_DIR, EXPERIMENT_FOLDER)

        print(f"\n✅ 第 {i+1} 次循環完美結束！")

        # ★ 防呆機制：讓系統休息 5 秒，確保 CARLA 徹底釋放記憶體與 Port 避免報錯
        if i < TOTAL_LOOPS - 1:
            print("⏳ 讓系統休息 5 秒鐘，準備進入下一次循環...")
            time.sleep(5)

    print(f"\n🎉 [完美收工] 設定的 {TOTAL_LOOPS} 次循環皆已全數自動執行完畢！")
    print("👉 你可以直接前往 inference_feedback_PLEM 資料夾查看最終的誤差數據了。")

if __name__ == "__main__":
    main()