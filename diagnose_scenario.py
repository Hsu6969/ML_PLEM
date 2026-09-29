# -*- coding: utf-8 -*-
"""
diagnose_scenario.py
對單一實驗資料夾做逐步體檢，找出 Y 車資料在哪一步掉的。
用法: python diagnose_scenario.py D:\CARLA_Experiments\20260929_xxxxxx
(Python 3.7 相容)
"""
import os
import sys
import math
import glob
import numpy as np
import pandas as pd


def read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print("   (讀取失敗: {})".format(e))
            return None
    return None


def meters(lat1, lon1, lat2, lon2):
    """小範圍近似距離 (公尺)"""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    dy = (lat2 - lat1) * 111320.0
    dx = (lon2 - lon1) * 111320.0 * np.cos(np.radians(lat1))
    return np.sqrt(dx ** 2 + dy ** 2)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_imu(folder):
    section("1. IMU 朝向 (compass)")
    for car in ("Z", "Y"):
        p = os.path.join(folder, "imu_{}.csv".format(car))
        if not os.path.exists(p):
            print("  {}: 找不到 {}".format(car, os.path.basename(p)))
            continue
        df = read_csv(p)
        if df is None or "orientation" not in df.columns:
            print("  {}: 無 orientation 欄位".format(car))
            continue
        o = df["orientation"].astype(float)
        print("  {}: n={}, 平均 {:.4f} rad = {:.1f} 度, 範圍 {:.4f}~{:.4f}".format(
            car, len(o), o.mean(), math.degrees(o.mean()), o.min(), o.max()))
    sc = os.path.join(folder, "scenario.txt")
    if os.path.exists(sc):
        with open(sc, encoding="utf-8") as f:
            print("  scenario.txt:", f.read().replace("\n", " | "))


def find_pred_cols(cols):
    lat = [c for c in cols if "lat" in c.lower() and ("pred" in c.lower() or "est" in c.lower())]
    lon = [c for c in cols if "lon" in c.lower() and ("pred" in c.lower() or "est" in c.lower())]
    return (lat[0], lon[0]) if lat and lon else (None, None)


def check_csvs(folder):
    section("2. 各 CSV 狀態 (列數 / inside==1 / gamma 範圍 / Δd)")
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
    for p in paths:
        rel = os.path.relpath(p, folder)
        df = read_csv(p)
        if df is None:
            continue
        cols = list(df.columns)
        info = ["rows={}".format(len(df))]

        if "inside/outside" in cols:
            ins = df[df["inside/outside"] == 1]
            info.append("inside==1: {}".format(len(ins)))
            if "p_NO" in cols and len(ins):
                cnt = ins.groupby("p_NO").size().to_dict()
                info.append("依行人: {}".format(cnt))

        if "gamma" in cols:
            g = df["gamma"].astype(float)
            flag = "  ⚠️ 超出 0~1" if (g.min() < -0.05 or g.max() > 1.05) else ""
            info.append("gamma(正規化) {:.3f}~{:.3f}{}".format(g.min(), g.max(), flag))

        if "predict_gamma" in cols:
            g = df["predict_gamma"].astype(float)
            info.append("predict_gamma {:.3f}~{:.3f}".format(g.min(), g.max()))

        plat, plon = find_pred_cols(cols)
        if plat and "p_lat" in cols and "p_lon" in cols:
            sub = df
            if "inside/outside" in cols:
                sub = df[df["inside/outside"] == 1]
            if len(sub):
                d = meters(sub["p_lat"], sub["p_lon"], sub[plat], sub[plon])
                info.append("Δd 平均 {:.2f} m / 中位 {:.2f} m / 最大 {:.2f} m".format(
                    np.nanmean(d), np.nanmedian(d), np.nanmax(d)))

        print("  {}\n      {}".format(rel, " | ".join(info)))


def check_yolo(folder):
    section("3. YOLO 偵測數 (labels 資料夾)")
    found = False
    for root, dirs, files in os.walk(folder):
        txts = [f for f in files if f.endswith(".txt") and f != "scenario.txt"]
        if not txts:
            continue
        found = True
        n_box, n_img_with = 0, 0
        for t in txts:
            try:
                with open(os.path.join(root, t)) as f:
                    lines = [l for l in f if l.strip()]
            except Exception:
                continue
            n_box += len(lines)
            n_img_with += 1 if lines else 0
        print("  {}: {} 個檔, 有偵測 {} 張, bbox 共 {} 個".format(
            os.path.relpath(root, folder), len(txts), n_img_with, n_box))
    for car in ("Z", "Y"):
        d = os.path.join(folder, "image_{}".format(car))
        if os.path.isdir(d):
            print("  image_{}: {} 張影像".format(car, len(os.listdir(d))))
    if not found:
        print("  (沒找到 YOLO label txt，可能輸出在別處)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_scenario.py <實驗資料夾>")
        sys.exit(1)
    folder = sys.argv[1]
    print("診斷資料夾:", folder)
    check_imu(folder)
    check_csvs(folder)
    check_yolo(folder)
    print("\n完成。請把 new 與舊 AB 資料夾兩份輸出一起貼上比對。")
