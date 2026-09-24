"""Q2 figures: affine transformation of a density, and Jensen's inequality for squaring."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

# --- left: Y = 1 + 3X applied to the exponential density (ties back to Q3c)
ax = axes[0]
t = np.linspace(-1, 14, 900)
fX = np.where(t < 0, 0, np.exp(-np.clip(t, 0, None)))
fY = np.where(t < 1, 0, np.exp(-np.clip(t - 1, 0, None) / 3) / 3)
ax.plot(t, fX, lw=2, label=r"$f_X(x)=e^{-x}$")
ax.plot(t, fY, lw=2, color="tab:red", label=r"$f_Y(y)=\frac{1}{3}e^{-(y-1)/3}$")
ax.fill_between(t, fX, alpha=.15)
ax.fill_between(t, fY, alpha=.15, color="tab:red")
ax.annotate("", xy=(1, 1/3), xytext=(0, 1),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="gray", ls="--"))
ax.text(.35, .72, "shift by $a$,\nsquash by $1/b$", fontsize=9, color="gray")
ax.set(title=r"Affine map $Y=a+bX$: area is conserved", xlabel="value",
       ylabel="density", xlim=(-1, 14), ylim=(0, 1.1))
ax.legend(); ax.grid(alpha=.3)

# --- right: Jensen's inequality, g(x) = x^2 on the sample [1,3,4]
ax = axes[1]
xs = np.array([1., 3., 4.])
xbar, m2 = xs.mean(), (xs**2).mean()
g = np.linspace(0, 5, 400)
ax.plot(g, g**2, lw=2, color="tab:blue", label=r"$g(x)=x^2$")
ax.plot(xs, xs**2, "o", ms=9, color="tab:blue", zorder=5)
for v in xs:
    ax.plot([v, v], [0, v**2], ls=":", lw=1, color="gray")
    ax.annotate(f"({v:.0f}, {v**2:.0f})", (v, v**2), textcoords="offset points",
                xytext=(8, -4), fontsize=9)
# the average of the three points sits above the curve
ax.plot([xbar], [m2], "s", ms=10, color="tab:red", zorder=6)
ax.plot([xbar], [xbar**2], "D", ms=9, color="tab:green", zorder=6)
ax.annotate("", xy=(xbar, m2), xytext=(xbar, xbar**2),
            arrowprops=dict(arrowstyle="<->", lw=2, color="black"))
ax.text(xbar + .12, (m2 + xbar**2) / 2, r"gap $=\mathrm{Var}(X)=\frac{14}{9}$",
        fontsize=10, va="center")
ax.axhline(m2, color="tab:red", ls="--", lw=1)
ax.axhline(xbar**2, color="tab:green", ls="--", lw=1)
ax.text(0.08, m2 + .35, r"$\overline{X^2}=26/3\approx8.67$", color="tab:red", fontsize=9.5)
ax.text(0.08, xbar**2 - 1.15, r"$\bar X^2=64/9\approx7.11$", color="tab:green", fontsize=9.5)
ax.plot([xbar], [0], "v", color="black", ms=7)
ax.text(xbar, -1.5, r"$\bar X=8/3$", ha="center", fontsize=9.5)
ax.set(title="Jensen: averaging then squaring $\\neq$ squaring then averaging",
       xlabel="x", ylabel="$x^2$", xlim=(0, 5.2), ylim=(-2.5, 20))
ax.legend(loc="upper left"); ax.grid(alpha=.3)

fig.tight_layout(); fig.savefig("solutions/q2_transformations.png", dpi=150)
