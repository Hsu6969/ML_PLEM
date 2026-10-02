r"""
merge_inspect_reid_features.py
==============================
把多輪實驗產生的 reid_features.csv 合併，並檢查資料量與特徵區別力。

用法 (可接在 auto_pipeline 後面自動執行)：
    python merge_inspect_reid_features.py D:\CARLA_Experiments
    python merge_inspect_reid_features.py D:\CARLA_Experiments --prefix 20260930 20261001
    python merge_inspect_reid_features.py D:\CARLA_Experiments --since 20260930_004000
    python merge_inspect_reid_features.py D:\CARLA_Experiments --exclude 20260930_002054 20260930_003153
    python merge_inspect_reid_features.py D:\CARLA_Experiments --since 20260930 --out D:\CARLA_Experiments\analysis_xxx

輪次篩選規則：
  - 只讀名稱為「YYYYMMDD_HHMMSS」的資料夾 (result_v1 這類會自動跳過)
  - --prefix : 資料夾名稱以這些字串開頭才讀 (可給多個)
  - --since  : 資料夾名稱 >= 這個值才讀 (時間戳記可直接比大小)
  - --exclude: 指定排除的資料夾
  - 根目錄若有 exclude_runs.txt (一行一個資料夾名稱，# 開頭為註解)，也會一併排除
    → 建議把校正輪次寫進這個檔案，以後就不會再被誤讀

需要套件：pandas、numpy、matplotlib
"""

import os
import re
import sys
import glob
import argparse
import numpy as np
import pandas as pd

FEATURES = ['f1_len_diff', 'f2_speed_diff', 'f3_mean_dist',
            'f4_dir_consistency', 'f5_acc_consistency']

EXPECTED = {
    'f1_len_diff': 'smaller',
    'f2_speed_diff': 'smaller',
    'f3_mean_dist': 'smaller',
    'f4_dir_consistency': 'larger',
    'f5_acc_consistency': 'smaller',
}

RUN_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def cohens_d(pos, neg):
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = pos.std(ddof=1), neg.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (pos.mean() - neg.mean()) / pooled


def load_exclude_file(root):
    p = os.path.join(root, 'exclude_runs.txt')
    out = set()
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.split('#')[0].strip()
                if line:
                    out.add(line)
    return out


def main(root, prefixes=None, since=None, exclude=None, out_dir=None):
    out_dir = out_dir or root
    os.makedirs(out_dir, exist_ok=True)
    exclude = set(exclude or []) | load_exclude_file(root)

    def run_folder(f):
        return os.path.relpath(f, root).split(os.sep)[0]

    def keep(name):
        if not RUN_PATTERN.match(name):
            return False
        if prefixes and not name.startswith(tuple(prefixes)):
            return False
        if since and name < since:
            return False
        return name not in exclude

    print("=" * 66)
    print("合併並檢查 reid_features.csv")
    print(f"根目錄: {root}")
    if prefixes:
        print(f"日期前綴: {', '.join(prefixes)}")
    if since:
        print(f"起始輪次: >= {since}")
    if exclude:
        print(f"排除輪次: {', '.join(sorted(exclude))}")
    print("=" * 66)

    all_files = sorted(glob.glob(os.path.join(root, '**', 'reid_features.csv'), recursive=True))
    files = [f for f in all_files if keep(run_folder(f))]

    if not files:
        print("❌ 找不到符合條件的 reid_features.csv。")
        return 1
    print(f"找到 {len(files)} 個 reid_features.csv\n")

    # ---- 合併 ----
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        src = run_folder(f)
        d['source'] = src
        sc = os.path.join(root, src, 'scenario.txt')
        if os.path.exists(sc):
            with open(sc, encoding='utf-8') as fh:
                d['scenario'] = fh.readline().strip()
        else:
            d['scenario'] = 'unknown'
        dfs.append(d)
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ['label'])

    out_csv = os.path.join(out_dir, 'reid_features_merged.csv')
    merged.to_csv(out_csv, index=False, encoding='utf-8-sig')

    pos = merged[merged['label'] == 1]
    neg = merged[merged['label'] == 0]
    n_total, n_pos, n_neg = len(merged), len(pos), len(neg)

    # ---- 每輪貢獻 ----
    print("每一輪各貢獻的樣本對：")
    per_src = merged.groupby(['source', 'scenario'])['label'].agg(['count', 'sum'])
    per_src.columns = ['總對數', '正樣本']
    per_src['負樣本'] = per_src['總對數'] - per_src['正樣本']
    print(per_src.to_string())

    # 提醒：正樣本不是 3 個的輪次
    odd = per_src[per_src['正樣本'] != 3]
    if len(odd):
        print(f"\n⚠️ 以下 {len(odd)} 輪正樣本不是 3 個 (可能有行人沒被兩車同時看到)：")
        print(odd.to_string())

    # ---- 分場景統計 ----
    print("\n各場景統計：")
    sc_tab = merged.groupby('scenario').agg(輪數=('source', 'nunique'),
                                            總對數=('label', 'count'),
                                            正樣本=('label', 'sum'))
    print(sc_tab.to_string())
    if 'unknown' in sc_tab.index:
        print("⚠️ 有輪次沒有 scenario.txt，請確認是否為改程式前跑的資料。")

    # ---- 總量摘要 ----
    print("\n" + "-" * 66)
    print("總量摘要：")
    print(f"  總樣本對: {n_total}   (正樣本 {n_pos} / 負樣本 {n_neg})")
    if n_pos > 0:
        print(f"  正:負 比例 ≈ 1 : {n_neg / n_pos:.1f}")
    if n_total < 100:
        level = "❌ 太少，只能看趨勢 (目標先衝到 100~200 對)"
    elif n_total < 500:
        level = "△ 可以開始初步訓練，但要更穩建議衝到 500 對以上"
    else:
        level = "✅ 量已足夠做 train/test 分割與正式訓練"
    print(f"  資料量評估: {level}")

    # ---- 特徵區別力 ----
    print("\n" + "-" * 66)
    print("各特徵區別力 (Cohen's d，絕對值越大越能分同一人/不同人)：")
    rows = []
    for feat in FEATURES:
        d = cohens_d(pos[feat], neg[feat])
        ok = ""
        if not np.isnan(d):
            pos_smaller = pos[feat].mean() < neg[feat].mean()
            ok = "✓方向符合" if pos_smaller == (EXPECTED[feat] == 'smaller') else "✗方向相反!"
        rows.append((feat, pos[feat].mean(), neg[feat].mean(), d, ok))
    rows.sort(key=lambda r: abs(r[3]) if not np.isnan(r[3]) else -1, reverse=True)
    print(f"  {'特徵':<20}{'正樣本均值':>12}{'負樣本均值':>12}{'|d|':>8}   判讀")
    for feat, mp, mn, d, ok in rows:
        dv = f"{abs(d):.2f}" if not np.isnan(d) else "  NA"
        print(f"  {feat:<20}{mp:>12.4f}{mn:>12.4f}{dv:>8}   {ok}")
    print("  (|d|: 0.2小 / 0.5中 / 0.8大)")

    # 正樣本 f3 異常大的輪次 (可能是未校正資料)
    pos_f3 = pos.groupby('source')['f3_mean_dist'].mean()
    sus = pos_f3[pos_f3 > 4.0]
    if len(sus):
        print(f"\n⚠️ 以下輪次正樣本平均 f3 > 4 m，可能是距離校正前處理的資料，請確認：")
        print(sus.round(2).to_string())

    # ---- 畫圖 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.ravel()
        for i, feat in enumerate(FEATURES):
            ax = axes[i]
            ax.boxplot([neg[feat].values, pos[feat].values],
                       labels=['diff (0)', 'same (1)'], showfliers=False)
            d = cohens_d(pos[feat], neg[feat])
            ax.set_title(f"{feat}\nCohen's d = {d:.2f}" if not np.isnan(d) else feat)
            ax.grid(axis='y', alpha=0.3)
        axes[-1].axis('off')
        fig.suptitle(f"Re-ID feature distributions  (pos={n_pos}, neg={n_neg})", fontsize=13)
        fig.tight_layout()
        out_png = os.path.join(out_dir, 'reid_feature_distributions.png')
        fig.savefig(out_png, dpi=120)
        print("\n" + "-" * 66)
        print(f"✅ 合併表已存: {out_csv}")
        print(f"✅ 分布圖已存: {out_png}")
    except ImportError:
        print(f"\n✅ 合併表已存: {out_csv}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=r"D:\CARLA_Experiments")
    ap.add_argument("--prefix", nargs="*", default=None, help="資料夾名稱前綴，可多個")
    ap.add_argument("--since", default=None, help="只讀資料夾名稱 >= 此值的輪次")
    ap.add_argument("--exclude", nargs="*", default=None, help="排除的資料夾名稱")
    ap.add_argument("--out", default=None, help="輸出資料夾 (預設為根目錄)")
    a = ap.parse_args()
    sys.exit(main(a.root, a.prefix, a.since, a.exclude, a.out))