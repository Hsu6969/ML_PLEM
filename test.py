import pandas as pd, glob, numpy as np

folder = r"D:\CARLA_Experiments\20260930_002054\inference_feedback_PLEM"
for f in sorted(glob.glob(folder + r"\*.csv")):
    d = pd.read_csv(f); d = d[d["inside/outside"] == 1]
    pdist = [c for c in d.columns if "pred" in c.lower() and "dist" in c.lower()][0]
    g_gt, g_pr = d["gamma"] * 90 - 45, d["predict_gamma"] * 90 - 45      # 度
    e_d = (d[pdist] - d["dist"]) * 50                                      # 公尺
    print(f.split("\\")[-1], "n=%d" % len(d),
          "| 真值距離 %.1f~%.1f m" % (d["dist"].min()*50, d["dist"].max()*50),
          "| 距離誤差 平均 %+.2f 絕對 %.2f m" % (e_d.mean(), e_d.abs().mean()),
          "| γ真值 %+.1f° 預測 %+.1f°" % (g_gt.mean(), g_pr.mean()))