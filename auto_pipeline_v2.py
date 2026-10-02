import subprocess
import os
import sys
import time
import csv

# ==================================================================
# ★ 基本設定
# ==================================================================
WORK_DIR = r"D:/ML_PLEM"
ENV_CARLA = "carla37"
ENV_YOLO = "yolov8_cu12"
ENV_ANALYSIS = "carla37"          # merge / evaluate 用的環境 (需有 sklearn、xgboost、matplotlib)
EXPERIMENT_ROOT = r"D:\CARLA_Experiments"

COLLECT_SCRIPT = "Precise_Vehicle_Placement_Random_v2.py"

# ★ 每個場景要跑幾輪 (場景代號必須存在於採集程式的 SCENARIOS 裡)
SCENARIO_ROUNDS = {
    "AB": 25,
    "AG": 25,
}

# True ：場景交錯跑 (AB, AG, AB, AG...)，中途 CARLA 掛掉時每個場景都已有資料
# False：一個場景跑完再跑下一個
INTERLEAVE = True

MAX_CONSECUTIVE_FAILURES = 3      # 同一場景連續失敗幾輪，就放棄該場景剩下的輪次
REST_SECONDS = 5                  # 每輪之間休息秒數，讓 CARLA 釋放資源

# 採集之後的 10 個步驟 (每輪都一樣)
POST_STEPS = [
    (ENV_CARLA, "clean_data.py"),                         # 2. 清理殘缺影像
    (ENV_CARLA, "concat_v2.py"),                          # 3. 串接 data_Z / data_Y
    (ENV_YOLO,  "predict.py"),                            # 4. YOLO 偵測
    (ENV_YOLO,  "merge_bbox_v4.py"),                      # 5. BBox 配對行人 ID
    (ENV_YOLO,  "convert_corner.py"),                     # 6. 座標轉左上/右下
    (ENV_YOLO,  "split_pedestrians.py"),                  # 7. 依行人切分
    (ENV_YOLO,  "create_time_window.py"),                 # 8. 4-slot 時間窗
    (ENV_YOLO,  "get_plem_features.py"),                  # 9. PLEM 特徵
    (ENV_YOLO,  "run_inference_plem.py"),                 # 10. PLEM 推論、反推 GPS
    (ENV_YOLO,  "build_reid_dataset_v2.py"),              # 11. 產生本輪 reid_features.csv
]


class StepFailed(Exception):
    pass


def run_script(env_name, script_name, args, log_file=None):
    """在指定 conda 環境執行腳本；失敗丟出 StepFailed。
       log_file 不為 None 時，輸出同步顯示並寫入 log。"""
    header = f"\n{'-' * 65}\n🚀 啟動: {script_name} {' '.join(args)} (環境: {env_name})\n"
    print(header, end="")
    if log_file is not None:
        log_file.write(header)
        log_file.flush()

    if log_file is None:
        cmd = ["conda.bat", "run", "--no-capture-output", "-n", env_name,
               "python", script_name] + list(args)
        rc = subprocess.run(cmd, cwd=WORK_DIR).returncode   # 不加 shell=True，確保等子程序結束
    else:
        cmd = ["conda.bat", "run", "--no-capture-output", "-n", env_name,
               "python", "-u", script_name] + list(args)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.Popen(cmd, cwd=WORK_DIR, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace")
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        rc = proc.wait()

    if rc != 0:
        raise StepFailed(f"{script_name} 執行失敗 (returncode={rc})")


def build_schedule():
    """依 SCENARIO_ROUNDS 排出執行順序。"""
    if not INTERLEAVE:
        return [sc for sc, n in SCENARIO_ROUNDS.items() for _ in range(n)]
    remaining = dict(SCENARIO_ROUNDS)
    schedule = []
    while any(v > 0 for v in remaining.values()):
        for sc in SCENARIO_ROUNDS:
            if remaining[sc] > 0:
                schedule.append(sc)
                remaining[sc] -= 1
    return schedule


def analyze_folder(folder, label, log):
    """對某個資料夾做 合併 -> 三類方法比較；結果存在該資料夾。"""
    banner = f"\n{'#' * 70}\n📊 分析 [{label}]  資料夾: {folder}\n{'#' * 70}\n"
    print(banner, end="")
    log.write(banner)
    try:
        run_script(ENV_ANALYSIS, "merge_inspect_reid_features.py", [folder], log)
        merged_csv = os.path.join(folder, "reid_features_merged.csv")
        if not os.path.exists(merged_csv):
            msg = f"⚠️ [{label}] 沒有產生 reid_features_merged.csv，略過方法比較。\n"
            print(msg, end="")
            log.write(msg)
            return False
        run_script(ENV_ANALYSIS, "evaluate_all_methods_3groups.py", [merged_csv], log)
        return True
    except StepFailed as e:
        msg = f"❌ [{label}] 分析失敗: {e}\n"
        print(msg, end="")
        log.write(msg)
        return False


def build_summary(run_folder, labels):
    """把各場景 (與全部合併) 的比較表整理成一張 scenario_summary.csv。"""
    rows = []
    for label in labels:
        folder = run_folder if label == "ALL" else os.path.join(run_folder, label)
        path = os.path.join(folder, "method_comparison_3groups.csv")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append({"scenario": label, "method": r["method"], "type": r["type"],
                             "acc": float(r["acc"]), "f1": float(r["f1"]),
                             "f1_std": float(r["f1_std"]), "auc": float(r["auc"])})
    if not rows:
        return None

    out = os.path.join(run_folder, "scenario_summary.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "method", "type", "acc", "f1", "f1_std", "auc"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'=' * 70}")
    print("各場景三類方法最佳結果 (F1)")
    print(f"{'=' * 70}")
    print(f"{'場景':<8}{'門檻值最佳':<26}{'傳統最佳':<22}{'ML最佳':<22}")
    for label in labels:
        sub = [r for r in rows if r["scenario"] == label]
        if not sub:
            continue
        cells = []
        for t in ["門檻值", "傳統", "ML"]:
            cand = [r for r in sub if r["type"] == t]
            if cand:
                b = max(cand, key=lambda r: r["f1"])
                cells.append(f"{b['method']} {b['f1']:.3f}")
            else:
                cells.append("-")
        print(f"{label:<8}{cells[0]:<26}{cells[1]:<22}{cells[2]:<22}")
    return out


def write_run_info(run_folder, start_ts, end_ts, results, skipped, analysis_status):
    path = os.path.join(run_folder, "run_info.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("本次實驗資訊\n" + "=" * 50 + "\n")
        f.write(f"開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_ts))}\n")
        f.write(f"結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_ts))}\n")
        f.write(f"總耗時: {(end_ts - start_ts) / 60:.1f} 分鐘\n")
        f.write(f"採集程式: {COLLECT_SCRIPT}\n")
        f.write(f"場景交錯執行: {INTERLEAVE}\n\n")
        for sc, planned in SCENARIO_ROUNDS.items():
            ok = results[sc]["ok"]
            fail = results[sc]["fail"]
            f.write(f"[{sc}] 預計 {planned} 輪 / 成功 {len(ok)} / 失敗 {len(fail)}"
                    f"{' (後段已放棄)' if sc in skipped else ''}\n")
            for r in ok:
                f.write(f"    ✓ {r}\n")
            for r, reason in fail:
                f.write(f"    ✗ {r}  ->  {reason}\n")
        f.write("\n分析狀態:\n")
        for label, ok in analysis_status.items():
            f.write(f"  {label}: {'完成' if ok else '未完成'}\n")
    return path


def main():
    start_ts = time.time()
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    RUN_FOLDER = os.path.join(EXPERIMENT_ROOT, f"run_{run_stamp}")
    os.makedirs(RUN_FOLDER, exist_ok=True)

    schedule = build_schedule()
    print(f"🎬 [全自動化管線啟動] 共 {len(schedule)} 輪："
          + "、".join(f"{sc} {n} 輪" for sc, n in SCENARIO_ROUNDS.items()))
    print(f"📁 本次實驗資料夾: {RUN_FOLDER}")

    results = {sc: {"ok": [], "fail": []} for sc in SCENARIO_ROUNDS}
    consecutive_fail = {sc: 0 for sc in SCENARIO_ROUNDS}
    skipped = set()

    for i, sc in enumerate(schedule):
        if sc in skipped:
            continue
        done_sc = len(results[sc]["ok"]) + len(results[sc]["fail"])
        print(f"\n{'=' * 70}")
        print(f"🔄 第 {i + 1} / {len(schedule)} 輪   場景 {sc} (第 {done_sc + 1} / {SCENARIO_ROUNDS[sc]} 輪)")
        print(f"{'=' * 70}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        EXPERIMENT_FOLDER = os.path.join(RUN_FOLDER, sc, timestamp)   # ★ run / 場景 / 輪次
        print(f"📁 本輪資料夾: {EXPERIMENT_FOLDER}")

        try:
            # 1. 採集：多傳 --scenario；資料夾放最後
            run_script(ENV_CARLA, COLLECT_SCRIPT, ["--scenario", sc, EXPERIMENT_FOLDER])
            # 2~11. 後處理
            for env_name, script_name in POST_STEPS:
                run_script(env_name, script_name, [EXPERIMENT_FOLDER])
            results[sc]["ok"].append(timestamp)
            consecutive_fail[sc] = 0
            print(f"\n✅ [{sc}] 本輪完成 (此場景成功 {len(results[sc]['ok'])} / 失敗 {len(results[sc]['fail'])})")
        except StepFailed as e:
            results[sc]["fail"].append((timestamp, str(e)))
            consecutive_fail[sc] += 1
            print(f"\n❌ [{sc}] 本輪失敗: {e}")
            print("   (此輪不會產生 reid_features.csv，合併時自動忽略，不會污染資料)")
            if consecutive_fail[sc] >= MAX_CONSECUTIVE_FAILURES:
                skipped.add(sc)
                print(f"⛔ [{sc}] 已連續失敗 {consecutive_fail[sc]} 輪，放棄此場景剩下的輪次。")

        if i < len(schedule) - 1:
            print(f"⏳ 休息 {REST_SECONDS} 秒，準備下一輪...")
            time.sleep(REST_SECONDS)

    # ---------- 分析：各場景各自一份 + 全部合併一份 ----------
    analysis_status = {}
    labels_done = []
    log_path = os.path.join(RUN_FOLDER, "analysis_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        for sc in SCENARIO_ROUNDS:
            if results[sc]["ok"]:
                analysis_status[sc] = analyze_folder(os.path.join(RUN_FOLDER, sc), sc, log)
                labels_done.append(sc)
            else:
                analysis_status[sc] = False
        scenarios_with_data = [sc for sc in SCENARIO_ROUNDS if results[sc]["ok"]]
        if len(scenarios_with_data) >= 2:
            analysis_status["ALL"] = analyze_folder(RUN_FOLDER, "ALL (全部場景合併)", log)
            labels_done.append("ALL")

    summary_path = build_summary(RUN_FOLDER, labels_done)
    write_run_info(RUN_FOLDER, start_ts, time.time(), results, skipped, analysis_status)

    print(f"\n{'=' * 70}")
    print("🎉 全部結束！")
    for sc in SCENARIO_ROUNDS:
        print(f"   [{sc}] 成功 {len(results[sc]['ok'])} 輪 / 失敗 {len(results[sc]['fail'])} 輪")
    print(f"📁 結果位置: {RUN_FOLDER}")
    print("   ├─ AB\\, AG\\ ...                     各場景的輪次資料夾 + 該場景的分析結果")
    print("   ├─ reid_features_merged.csv        全部場景合併資料")
    print("   ├─ method_comparison_3groups.csv   全部場景合併的三類方法比較")
    if summary_path:
        print("   ├─ scenario_summary.csv            各場景結果彙整表")
    print("   ├─ analysis_log.txt                分析階段完整輸出 (含 Cohen's d)")
    print("   └─ run_info.txt                    本次實驗摘要")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()