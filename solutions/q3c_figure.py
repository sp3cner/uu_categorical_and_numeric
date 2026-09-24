"""Q3c: F_X vs F_Y for Y = 1 + 3X, with Pr[Y < 10] read off the curve."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.5, 5))
t = np.linspace(-1, 17, 1200)
FX = np.where(t < 0, 0, 1 - np.exp(-np.clip(t, 0, None)))
FY = np.where(t < 1, 0, 1 - np.exp(-np.clip(t - 1, 0, None) / 3))

ax.plot(t, FX, lw=2.2, label=r"$F_X(x)=1-e^{-x}$  (starts at 0)")
ax.plot(t, FY, lw=2.2, color="tab:red", label=r"$F_Y(y)=1-e^{-(y-1)/3}$  (starts at 1)")
ax.axhline(1, color="gray", ls="--", lw=1)

p = 1 - np.exp(-3)
ax.plot([10, 10], [0, p], color="k", ls=":", lw=1.4)
ax.plot([-1, 10], [p, p], color="k", ls=":", lw=1.4)
ax.plot([10], [p], "o", color="k", ms=8, zorder=5)
ax.annotate(r"$\Pr[Y<10]=1-e^{-3}\approx0.950$", xy=(10, p), xytext=(10.6, 0.72),
            fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.2))
ax.plot([3], [p], "o", color="tab:blue", ms=8, zorder=5)
ax.annotate(r"same height: $\Pr[X<3]$", xy=(3, p), xytext=(3.3, 0.55),
            fontsize=10, color="tab:blue",
            arrowprops=dict(arrowstyle="->", lw=1.1, color="tab:blue"))

ax.annotate("", xy=(1, 0.02), xytext=(0, 0.02),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="gray"))
ax.text(0.05, 0.065, "shift +1", fontsize=9, color="gray")

ax.set(xlabel="value", ylabel="cumulative probability",
       title=r"Q3c: $Y=1+3X$ shifts right by 1 and stretches by 3",
       xlim=(-1, 17), ylim=(0, 1.08))
ax.set_yticks([0, .25, .5, .75, .95, 1])
ax.legend(loc="lower right", fontsize=10); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("solutions/q3c_comparison.png", dpi=150)
