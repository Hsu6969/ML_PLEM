"""
batch_rebuild_reid.py
=====================
自動掃過根目錄底下所有實驗資料夾，逐一重跑 build_reid_dataset.py，
把每個資料夾的 reid_features.csv 重新產生成「含 ED/DTW/LCSS 傳統方法欄位」的新版。

判斷依據：資料夾裡有 inference_feedback_PLEM/ 就視為一個有效的實驗輪次。
只讀 inference_GPS_*.csv，不需要重開 CARLA，很快。

用法：
    python batch_rebuild_reid.py  D:\CARLA_Experiments
若不給參數則用預設根目錄。

前提：batch_rebuild_reid.py 要跟 build_reid_dataset_v2.py 放在同一個資料夾。
"""

import os
import sys
import glob
import importlib.util


def load_build_module():
    """載入同資料夾下的 build_reid_dataset_v2.py，取得它的 build_dataset 函式。"""
    here = os.path.dirname(os.path.abspath(__file__))
    build_path = os.path.join(here, "build_reid_dataset_v2.py")
    if not os.path.exists(build_path):
        print(f"❌ 找不到 build_reid_dataset_v2.py，請把它跟本腳本放在同一資料夾: {here}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("build_reid_dataset_v2", build_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(root):
    print("=" * 66)
    print("批次重建 reid_features.csv (補上 ED/DTW/LCSS 欄位)")
    print(f"根目錄: {root}")
    print("=" * 66)

    if not os.path.isdir(root):
        print(f"❌ 根目錄不存在: {root}")
        return

    build = load_build_module()

    # 找出所有「含 inference_feedback_PLEM 資料夾」的實驗資料夾
    marker_dirs = glob.glob(os.path.join(root, "*", "inference_feedback_PLEM"))
    exp_folders = sorted(os.path.dirname(m) for m in marker_dirs)

    if not exp_folders:
        print("❌ 根目錄下找不到任何含 inference_feedback_PLEM 的實驗資料夾。")
        return

    print(f"找到 {len(exp_folders)} 個實驗資料夾，開始逐一重建...\n")

    ok, fail, empty = 0, 0, 0
    for i, folder in enumerate(exp_folders, 1):
        name = os.path.basename(folder)
        print(f"[{i}/{len(exp_folders)}] {name}")
        try:
            build.build_dataset(folder)
            # 確認有沒有真的產生檔案 (重疊不足時 build_dataset 不會寫檔)
            if os.path.exists(os.path.join(folder, "reid_features.csv")):
                ok += 1
            else:
                empty += 1
        except Exception as e:
            print(f"   ❌ 這個資料夾處理失敗: {e}")
            fail += 1
        print()

    print("=" * 66)
    print("批次完成摘要：")
    print(f"  成功重建: {ok} 個")
    print(f"  無有效樣本 (重疊不足，未產生檔案): {empty} 個")
    print(f"  處理失敗: {fail} 個")
    print(f"  總資料夾: {len(exp_folders)} 個")
    print("\n下一步：跑 merge_inspect_reid_features.py 重新合併，")
    print("        合併表就會同時含 f1~f5 與 ed_dist / dtw_dist / lcss_sim。")
    print("=" * 66)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\CARLA_Experiments"
    main(root)
