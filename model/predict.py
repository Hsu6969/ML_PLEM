from ultralytics import YOLO
import os
import sys

#將Precise_Vehicle_Placement_V5_2拍攝的圖片讀取經過yolov8模型生成bounding box

def predict_by_YOLO(image_path, save_directory):

    # model=YOLO("yolov8m.pt")
    # model = YOLO("all_train.pt") #偵測包含行人和車輛 用來偵測車輛

    # model = YOLO("all_train_only_ped.pt") #只偵測行人 
    model = YOLO("ped_0321.pt") #只偵測行人 0321新train
    # model = YOLO("john_yolov5.pt") #學長偵測車輛

    os.makedirs(save_directory, exist_ok=True)

    model.iou=0.5 # 學長車輛YOLO設定 0.5#TODO
    model.conf=0.8 # 學長車輛YOLO設定 0.7

    """
    save=True,"存預測下來的圖片"
    """
    result = model.predict(
        source=image_path,
        mode="predict",
        save=True,
        save_txt=True,
        show_labels=False,
        project=save_directory
    )
    """
    不存圖片
    """
    print("YOLO8 predict完成")

# if __name__ == "__main__":

#     z_image_path=r'D:/CARLA_Experiments/20260311_185919/image_Z'
#     save_directory = r'D:/CARLA_Experiments/20260311_185919/image_Z/predict_result'
#     predict_by_YOLO(z_image_path, save_directory)
#     y_image_path=r'D:/CARLA_Experiments/20260311_185919/image_Y'
#     save_directory = r'D:/CARLA_Experiments/20260311_185919/image_Y/predict_result'
#     predict_by_YOLO(y_image_path, save_directory)

if __name__ == "__main__":
    # ==============================================================
    # ★ 自動化管線接收參數修改
    # ==============================================================
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
        print(f"🔗 [YOLO預測] 接收到指定資料夾路徑: {base_folder}")
    else:
        # 手動執行時的預設路徑 (方便你單獨測試)
        base_folder = r'D:\CARLA_Experiments\default_test'
        print(f"⏰ [手動執行] 使用預設資料夾路徑: {base_folder}")

    # --------------------------------------------------
    # 1. 處理 Z 車的照片
    # --------------------------------------------------
    z_image_path = os.path.join(base_folder, 'image_Z')
    z_save_directory = os.path.join(z_image_path, 'predict_result')
    
    print(f"🔄 正在處理 Z 車照片...")
    predict_by_YOLO(z_image_path, z_save_directory)

    # --------------------------------------------------
    # 2. 處理 Y 車的照片
    # --------------------------------------------------
    y_image_path = os.path.join(base_folder, 'image_Y')
    y_save_directory = os.path.join(y_image_path, 'predict_result')
    
    print(f"🔄 正在處理 Y 車照片...")
    predict_by_YOLO(y_image_path, y_save_directory)
    
    print("✅ YOLO 預測全部完成！")

"""
save=True  保存檢測後輸出的圖像
conf default=0.25
iou default=0.7
save_txt 識別結果存為txt
hide_label 保存識別圖像是否隱藏label
line_thickness 目標框中的線條粗細 default=3
augment 是否使用數據增強 default=false
"""