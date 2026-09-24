"""Q5: CDF and quantile function are reflections across the 45-degree line."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

# ---- exponential
ax = axes[0]
x = np.linspace(0, 4.2, 600)
u = np.linspace(1e-4, 0.985, 600)
ax.plot(x, 1 - np.exp(-x), lw=2.2, label=r"CDF  $F(x)=1-e^{-x}$")
ax.plot(u, -np.log(1 - u), lw=2.2, color="tab:red",
        label=r"quantile  $F^{-1}(u)=-\ln(1-u)$")
ax.plot([0, 4.2], [0, 4.2], color="gray", ls="--", lw=1.2, label=r"$y=x$ (mirror)")
m = np.log(2)
ax.plot([m], [.5], "o", color="tab:blue", ms=9, zorder=5)
ax.plot([.5], [m], "o", color="tab:red", ms=9, zorder=5)
ax.annotate(r"$F(\ln 2)=0.5$", xy=(m, .5), xytext=(1.15, .34), fontsize=10,
            color="tab:blue", arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.1))
ax.annotate(r"$F^{-1}(0.5)=\ln 2\approx0.693$", xy=(.5, m), xytext=(.12, 1.75),
            fontsize=10, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.1))
ax.set(title="Exponential: median $=\\ln 2 \\approx 0.693$",
       xlabel="input", ylabel="output", xlim=(0, 4.2), ylim=(0, 4.2))
ax.legend(loc="lower right", fontsize=9.5); ax.grid(alpha=.3); ax.set_aspect("equal")

# ---- logistic
ax = axes[1]
x = np.linspace(-4.5, 4.5, 600)
u = np.linspace(0.012, 0.988, 600)
ax.plot(x, 1 / (1 + np.exp(-x)), lw=2.2, label=r"CDF  $F(x)=\frac{1}{1+e^{-x}}$")
ax.plot(u, np.log(u / (1 - u)), lw=2.2, color="tab:red",
        label=r"quantile  $F^{-1}(u)=\ln\frac{u}{1-u}$")
ax.plot([-4.5, 4.5], [-4.5, 4.5], color="gray", ls="--", lw=1.2, label=r"$y=x$ (mirror)")
ax.plot([0], [.5], "o", color="tab:blue", ms=9, zorder=5)
ax.plot([.5], [0], "o", color="tab:red", ms=9, zorder=5)
ax.annotate(r"$F(0)=0.5$", xy=(0, .5), xytext=(.55, 1.35), fontsize=10,
            color="tab:blue", arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.1))
ax.annotate(r"$F^{-1}(0.5)=0$", xy=(.5, 0), xytext=(1.25, -1.5), fontsize=10,
            color="tab:red", arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.1))
ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="k", lw=.8)
ax.set(title="Logistic: median $=0$", xlabel="input", ylabel="output",
       xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
ax.legend(loc="lower right", fontsize=9.5); ax.grid(alpha=.3); ax.set_aspect("equal")

fig.tight_layout(); fig.savefig("solutions/q5_quantiles.png", dpi=150)
