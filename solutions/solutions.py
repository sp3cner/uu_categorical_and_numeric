"""Figures for the Categorical and Numeric problem set (Q1c, Q3, Q4, Q6)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "solutions"

# ---------------------------------------------------------------- Q1c: ECDF of X = [1,3,3,6,11]
x = np.array([1, 3, 3, 6, 11])
fig, ax = plt.subplots(figsize=(6, 4))
grid = np.sort(np.unique(x))
ecdf = np.array([(x <= g).mean() for g in grid])
xs = np.concatenate([[-1], grid, [13]])
ys = np.concatenate([[0], ecdf, [1]])
ax.step(xs, ys, where="post", lw=2)
ax.plot(grid, ecdf, "o", ms=7)                      # closed: F is right-continuous
ax.plot(grid, np.concatenate([[0], ecdf[:-1]]), "o", mfc="white", ms=7)
ax.set(xlabel="x", ylabel=r"$\hat F(x)$", title="Q1c: ECDF of X = [1, 3, 3, 6, 11]",
       ylim=(-0.05, 1.05))
ax.set_yticks([0, .2, .4, .6, .8, 1])
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/q1c_ecdf.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- Q3: exponential CDF/pdf, and Y = 1 + 3X
t = np.linspace(-1, 6, 700)
F = np.where(t < 0, 0, 1 - np.exp(-np.clip(t, 0, None)))
f = np.where(t < 0, 0, np.exp(-np.clip(t, 0, None)))
y = np.linspace(-1, 16, 700)
FY = np.where(y < 1, 0, 1 - np.exp(-(y - 1) / 3))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].plot(t, F, lw=2); axes[0].set(title=r"Q3a: $F_X(x)=1-e^{-x}$", xlabel="x", ylabel="F(x)")
axes[1].plot(t, f, lw=2); axes[1].set(title=r"Q3a: $f_X(x)=e^{-x}$", xlabel="x", ylabel="f(x)")
axes[2].plot(t, F, lw=2, label=r"$F_X$")
axes[2].plot(y, FY, lw=2, label=r"$F_Y,\ Y=1+3X$")
axes[2].axvline(10, color="k", ls=":", lw=1)
axes[2].set(title="Q3c: stretched and shifted", xlabel="value", ylabel="CDF", xlim=(-1, 16))
axes[2].legend()
for a in axes: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/q3_exponential.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- Q4: logistic CDF/pdf, and Y = 2X - 1
t = np.linspace(-8, 8, 800)
F = 1 / (1 + np.exp(-t))
f = F * (1 - F)
y = np.linspace(-14, 14, 800)
FY = 1 / (1 + np.exp(-(y + 1) / 2))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].plot(t, F, lw=2); axes[0].set(title=r"Q4a: $F_X(x)=1/(1+e^{-x})$", xlabel="x", ylabel="F(x)")
axes[1].plot(t, f, lw=2); axes[1].set(title=r"Q4a: $f_X(x)=F(1-F)$", xlabel="x", ylabel="f(x)")
axes[2].plot(t, F, lw=2, label=r"$F_X$")
axes[2].plot(y, FY, lw=2, label=r"$F_Y,\ Y=2X-1$")
axes[2].axvline(0, color="k", ls=":", lw=1)
axes[2].set(title="Q4c: wider, centered at -1", xlabel="value", ylabel="CDF")
axes[2].legend()
for a in axes: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/q4_logistic.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- Q6: METABRIC ECDFs
df = pd.read_csv("data/metabric.csv")
col = "Overall Survival (Months)"

def ecdf_xy(v):
    v = np.sort(np.asarray(v.dropna()))
    return v, np.arange(1, v.size + 1) / v.size

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
xv, yv = ecdf_xy(df[col])
axes[0].step(xv, yv, where="post", lw=2)
axes[0].set(title="Q6a: ECDF of overall survival", xlabel=col, ylabel="proportion of patients")
for lab, color in [("NO", "tab:blue"), ("YES", "tab:red")]:
    xv, yv = ecdf_xy(df.loc[df["Chemotherapy"] == lab, col])
    axes[1].step(xv, yv, where="post", lw=2, color=color,
                 label=f"Chemotherapy = {lab} (n={xv.size})")
axes[1].set(title="Q6b: ECDF by chemotherapy", xlabel=col, ylabel="proportion of patients")
axes[1].legend()
for a in axes:
    a.grid(alpha=.3); a.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/q6_metabric_ecdf.png", dpi=150); plt.close(fig)

# numbers quoted in the write-up
g = df.groupby("Chemotherapy")[col]
print(g.agg(count="count", mean="mean", median="median",
            q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75)).round(1))
print("\nP(survival > t):")
print(pd.DataFrame({t: g.apply(lambda s, t=t: (s > t).mean()) for t in (50, 100, 150, 200)}).round(3))
print("\nconfounders (medians):")
print(df.groupby("Chemotherapy")[["Age at Diagnosis", "Tumor Size",
                                  "Lymph nodes examined positive",
                                  "Nottingham prognostic index"]].median().round(2))
