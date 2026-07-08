"""
merge_inspect_reid_features.py
==============================
把多輪實驗產生的 reid_features.csv 全部合併，並檢查資料是否足夠、特徵有沒有區別力。

它會：
  1. 遞迴掃描根目錄下所有 reid_features.csv (每輪一個)，合併成一張大表。
  2. 統計總樣本對數、正/負樣本數與比例，以及每一輪各貢獻幾對。
  3. 對每個特徵計算 Cohen's d (標準化的正負平均差)，數值越大代表越能區分同一人/不同人。
  4. 畫出每個特徵在「同一人 vs 不同人」的箱型圖，存成 PNG。

用法：
    python merge_inspect_reid_features.py  D:\CARLA_Experiments
(不給參數則用預設根目錄)

需要套件：pandas、numpy、matplotlib
    pip install pandas numpy matplotlib
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

FEATURES = ['f1_len_diff', 'f2_speed_diff', 'f3_mean_dist',
            'f4_dir_consistency', 'f5_acc_consistency']

# 每個特徵「同一人(正樣本)」預期的方向：
#   smaller = 正樣本應該比較小 (距離/差異類)
#   larger  = 正樣本應該比較大 (方向一致性)
EXPECTED = {
    'f1_len_diff': 'smaller',
    'f2_speed_diff': 'smaller',
    'f3_mean_dist': 'smaller',
    'f4_dir_consistency': 'larger',
    'f5_acc_consistency': 'smaller',
}


def cohens_d(pos, neg):
    """標準化平均差；|d| 越大越能分開兩群 (0.2小 / 0.5中 / 0.8大)。"""
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = pos.std(ddof=1), neg.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (pos.mean() - neg.mean()) / pooled


def main(root):
    files = sorted(glob.glob(os.path.join(root, '**', 'reid_features.csv'), recursive=True))

    print("=" * 66)
    print("合併並檢查 reid_features.csv")
    print(f"根目錄: {root}")
    print("=" * 66)

    if not files:
        print("❌ 找不到任何 reid_features.csv，請確認先跑過 build_reid_dataset.py。")
        return

    print(f"找到 {len(files)} 個 reid_features.csv\n")

    # ---- 合併 ----
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d['source'] = os.path.basename(os.path.dirname(f))   # 用資料夾名標記來源輪次
        dfs.append(d)
    merged = pd.concat(dfs, ignore_index=True)

    # 清掉可能的 NaN / inf
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ['label'])

    out_csv = os.path.join(root, 'reid_features_merged.csv')
    merged.to_csv(out_csv, index=False, encoding='utf-8-sig')

    pos = merged[merged['label'] == 1]
    neg = merged[merged['label'] == 0]
    n_total, n_pos, n_neg = len(merged), len(pos), len(neg)

    # ---- 每輪貢獻 ----
    print("每一輪各貢獻的樣本對：")
    per_src = merged.groupby('source')['label'].agg(['count', 'sum'])
    per_src.columns = ['總對數', '正樣本']
    per_src['負樣本'] = per_src['總對數'] - per_src['正樣本']
    print(per_src.to_string())

    # ---- 總量摘要 ----
    print("\n" + "-" * 66)
    print("總量摘要：")
    print(f"  總樣本對: {n_total}   (正樣本 {n_pos} / 負樣本 {n_neg})")
    if n_pos > 0:
        print(f"  正:負 比例 ≈ 1 : {n_neg / n_pos:.1f}")

    if n_total < 100:
        level = "❌ 太少，只能看趨勢，還不能認真訓練 (目標先衝到 100~200 對)"
    elif n_total < 500:
        level = "△ 可以開始初步訓練看看，但要更穩建議衝到 500 對以上"
    else:
        level = "✅ 量已足夠做 train/test 分割與正式訓練"
    print(f"  資料量評估: {level}")

    # ---- 特徵區別力 (Cohen's d) ----
    print("\n" + "-" * 66)
    print("各特徵區別力 (Cohen's d，絕對值越大越能分同一人/不同人)：")
    rows = []
    for feat in FEATURES:
        d = cohens_d(pos[feat], neg[feat])
        # 檢查方向對不對 (正樣本是否如預期比較小/大)
        direction_ok = ""
        if not np.isnan(d):
            pos_smaller = pos[feat].mean() < neg[feat].mean()
            want_smaller = (EXPECTED[feat] == 'smaller')
            direction_ok = "✓方向符合" if pos_smaller == want_smaller else "✗方向相反!"
        rows.append((feat, pos[feat].mean(), neg[feat].mean(), d, direction_ok))

    rows.sort(key=lambda r: abs(r[3]) if not np.isnan(r[3]) else -1, reverse=True)
    print(f"  {'特徵':<20}{'正樣本均值':>12}{'負樣本均值':>12}{'|d|':>8}   判讀")
    for feat, mp, mn, d, ok in rows:
        dv = f"{abs(d):.2f}" if not np.isnan(d) else "  NA"
        print(f"  {feat:<20}{mp:>12.4f}{mn:>12.4f}{dv:>8}   {ok}")
    print("  (|d|: 0.2小 / 0.5中 / 0.8大；越大越有用。方向相反代表該特徵可能沒幫助或算法要檢查)")

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
        axes[-1].axis('off')   # 第 6 格空著
        fig.suptitle(f"Re-ID feature distributions  (pos={n_pos}, neg={n_neg})", fontsize=13)
        fig.tight_layout()
        out_png = os.path.join(root, 'reid_feature_distributions.png')
        fig.savefig(out_png, dpi=120)
        print("\n" + "-" * 66)
        print(f"✅ 合併表已存: {out_csv}")
        print(f"✅ 分布圖已存: {out_png}")
    except ImportError:
        print("\n⚠️ 未安裝 matplotlib，略過畫圖 (pip install matplotlib 後可產生分布圖)。")
        print(f"✅ 合併表已存: {out_csv}")

    print("=" * 66)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\CARLA_Experiments"
    main(root)