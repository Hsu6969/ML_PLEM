import pandas as pd, glob, numpy as np
folders = [r"D:\CARLA_Experiments\20260930_003153", r"D:\CARLA_Experiments\20260930_002054", r"D:\CARLA_Experiments\20260803_063652"]
ratios = []
for fd in folders:
    for f in glob.glob(fd + r"\inference_feedback_PLEM\inference_GPS_*_P*.csv"):
        d = pd.read_csv(f); d = d[d["inside/outside"] == 1]
        pc = [c for c in d.columns if "pred" in c.lower() and "dist" in c.lower()][0]
        ratios.append((d[pc] / d["dist"]).values)
r = np.concatenate(ratios)
boot = [np.median(np.random.default_rng(i).choice(r, len(r))) for i in range(2000)]
print(f"n={len(r)}  合併中位數 k = {np.median(r):.3f}  95% CI [{np.percentile(boot,2.5):.3f}, {np.percentile(boot,97.5):.3f}]")