"""Q4c: logistic X vs Y = 2X - 1, CDFs and densities."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = np.linspace(-12, 10, 1400)
FX = 1 / (1 + np.exp(-t))
FY = 1 / (1 + np.exp(-(t + 1) / 2))
fX = FX * (1 - FX)
fY = 0.5 * FY * (1 - FY)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

ax = axes[0]
ax.plot(t, FX, lw=2.2, label=r"$F_X(x)=\frac{1}{1+e^{-x}}$")
ax.plot(t, FY, lw=2.2, color="tab:red", label=r"$F_Y(y)=\frac{1}{1+e^{-(y+1)/2}}$")
p = 1 / (1 + np.exp(-0.5))
ax.plot([0, 0], [0, p], "k:", lw=1.4)
ax.plot([-12, 0], [p, p], "k:", lw=1.4)
ax.plot([0], [p], "o", color="k", ms=8, zorder=5)
ax.annotate(r"$\Pr[Y<0]\approx0.622$", xy=(0, p), xytext=(1.0, 0.45),
            fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.2))
ax.plot([0.5], [p], "o", color="tab:blue", ms=8, zorder=5)
ax.annotate(r"same height: $\Pr[X<0.5]$", xy=(0.5, p), xytext=(1.6, 0.85),
            fontsize=10, color="tab:blue",
            arrowprops=dict(arrowstyle="->", lw=1.1, color="tab:blue"))
ax.axhline(.5, color="gray", ls="--", lw=1)
ax.plot([0, -1], [.5, .5], "o", ms=6, color="gray")
ax.text(-5.6, .53, "centers: $X$ at 0, $Y$ at $-1$", fontsize=9, color="gray")
ax.set(title="Q4c: CDFs — same S-curve, recentered and widened",
       xlabel="value", ylabel="cumulative probability", ylim=(0, 1.05))
ax.legend(loc="upper left", fontsize=9.5); ax.grid(alpha=.3)

ax = axes[1]
ax.plot(t, fX, lw=2.2, label=r"$f_X$ (peak $1/4$ at $0$)")
ax.plot(t, fY, lw=2.2, color="tab:red", label=r"$f_Y$ (peak $1/8$ at $-1$)")
ax.fill_between(t, fX, alpha=.15)
ax.fill_between(t, fY, alpha=.15, color="tab:red")
ax.set(title="Densities: twice as wide, half as tall (area still 1)",
       xlabel="value", ylabel="density", ylim=(0, .29))
ax.legend(fontsize=9.5); ax.grid(alpha=.3)

fig.tight_layout(); fig.savefig("solutions/q4c_comparison.png", dpi=150)
