# Solutions: Categorical and Numeric Variables

Figures are produced by [`solutions/solutions.py`](solutions/solutions.py).

---

## Q1

### a. One-hot encoding

Data: `[red, blue, blue, red, green, grey]`. Four levels → four indicator columns (alphabetical order).

| obs | blue | green | grey | red |
|-----|------|-------|------|-----|
| 1 (red)   | 0 | 0 | 0 | 1 |
| 2 (blue)  | 1 | 0 | 0 | 0 |
| 3 (blue)  | 1 | 0 | 0 | 0 |
| 4 (red)   | 0 | 0 | 0 | 1 |
| 5 (green) | 0 | 1 | 0 | 0 |
| 6 (grey)  | 0 | 0 | 1 | 0 |

$$
X = \begin{bmatrix}
0&0&0&1\\ 1&0&0&0\\ 1&0&0&0\\ 0&0&0&1\\ 0&1&0&0\\ 0&0&1&0
\end{bmatrix}
$$

Every row sums to 1 — the columns are linearly dependent (they sum to the all-ones vector), which is why you drop one column when you put this into a regression with an intercept.

### b. Sample proportions

The sample proportion of a level is just the column mean of its indicator: $\hat p_k = \frac{1}{n}\sum_i \mathbb{1}[x_i = k]$.

| label | count | proportion |
|-------|-------|------------|
| red   | 2 | 2/6 = 1/3 ≈ 0.333 |
| blue  | 2 | 2/6 = 1/3 ≈ 0.333 |
| green | 1 | 1/6 ≈ 0.167 |
| grey  | 1 | 1/6 ≈ 0.167 |

Sums to 1. ✓

### c. ECDF of $X = [1, 3, 3, 6, 11]$

$\hat F(x) = \frac{1}{n}\sum_i \mathbb{1}[x_i \le x]$ with $n = 5$: a right-continuous step function that jumps by $1/5$ at each observation, and by $2/5$ at $x=3$ because that value is repeated.

$$
\hat F(x) = \begin{cases}
0,   & x < 1\\
0.2, & 1 \le x < 3\\
0.6, & 3 \le x < 6\\
0.8, & 6 \le x < 11\\
1,   & x \ge 11
\end{cases}
$$

![Q1c ECDF](solutions/q1c_ecdf.png)

---

## Q2

### a. Distribution and density of $Y = a + bX$, $b > 0$

$$
F_Y(y) = \Pr[Y \le y] = \Pr[a + bX \le y] = \Pr\!\left[X \le \tfrac{y-a}{b}\right] = F_X\!\left(\frac{y-a}{b}\right)
$$

(The inequality direction is preserved only because $b > 0$; for $b<0$ it flips and you get $1 - F_X(\frac{y-a}{b})$ for continuous $X$.)

Differentiate with the chain rule:

$$
f_Y(y) = \frac{d}{dy} F_X\!\left(\frac{y-a}{b}\right) = \frac{1}{b}\, f_X\!\left(\frac{y-a}{b}\right)
$$

So an affine map shifts the distribution by $a$, stretches it by $b$, and scales the density down by $1/b$ so it still integrates to 1. Mean and SD transform predictably: $E[Y] = a + bE[X]$, $\mathrm{SD}(Y) = b\,\mathrm{SD}(X)$.

### b. Mean of $X = [1, 3, 4]$

$$\bar X = \frac{1+3+4}{3} = \frac{8}{3} \approx 2.667$$

### c. Mean of $X^2$

$X^2 = [1, 9, 16]$:

$$\overline{X^2} = \frac{1+9+16}{3} = \frac{26}{3} \approx 8.667$$

### d. Square of the mean

$$\bar X^2 = \left(\frac{8}{3}\right)^2 = \frac{64}{9} \approx 7.111$$

**No** — $7.111 \ne 8.667$. The mean of the square exceeds the square of the mean. The gap is exactly the (population) variance:

$$\overline{X^2} - \bar X^2 = \frac{78}{9} - \frac{64}{9} = \frac{14}{9} \approx 1.556 = \widehat{\mathrm{Var}}(X)$$

This is Jensen's inequality: for a convex function $g$, $\overline{g(X)} \ge g(\bar X)$. Averaging and non-linear transformation do not commute — so log-transforming a feature and then averaging gives something different from averaging then logging.

---

## Q3 — Exponential distribution

### a. Sketch

$F(x) = 1 - e^{-x}$ for $x \ge 0$, so the density is

$$f(x) = F'(x) = \begin{cases} e^{-x}, & x \ge 0\\ 0, & x<0\end{cases}$$

The CDF starts at 0, rises steeply, and flattens asymptotically toward 1. The density starts at its maximum $f(0)=1$ and decays monotonically — no mode in the interior, heavy-ish right tail.

![Q3 exponential](solutions/q3_exponential.png)

### b. Tail probabilities

$$\Pr[X \ge 3] = 1 - F(3) = e^{-3} \approx 0.0498$$

$$\Pr[X \le 35] = F(35) = 1 - e^{-35} \approx 1 - 6.3\times10^{-16} \approx 1.000$$

### c. $Y = 1 + 3X$

Using Q2a with $a=1$, $b=3$:

$$
F_Y(y) = F_X\!\left(\frac{y-1}{3}\right) = \begin{cases}
0, & y < 1\\
1 - e^{-(y-1)/3}, & y \ge 1
\end{cases}
$$

Density: $f_Y(y) = \frac{1}{3}e^{-(y-1)/3}$ for $y \ge 1$. This is exponential with rate $1/3$, shifted to start at 1 — the curve is the same shape as $F_X$ but pushed right by 1 and stretched horizontally by a factor of 3, so it approaches 1 more slowly (right panel of the figure above).

$$\Pr[Y < 10] = 1 - e^{-(10-1)/3} = 1 - e^{-3} \approx 0.9502$$

Note this equals $\Pr[X < 3]$ — the event $\{Y<10\}$ *is* the event $\{X<3\}$. Monotone transformations relabel the axis but do not change probabilities.

---

## Q4 — Logistic distribution

### a. Sketch

$F(x) = \dfrac{1}{1+e^{-x}}$, so

$$f(x) = F'(x) = \frac{e^{-x}}{(1+e^{-x})^2} = F(x)\big(1 - F(x)\big)$$

The CDF is the familiar S-curve on all of $\mathbb{R}$, with $F(0) = 1/2$. The density is symmetric and bell-shaped, peaking at $f(0) = 1/4$, with tails heavier than a normal's.

![Q4 logistic](solutions/q4_logistic.png)

### b. Probabilities

$$\Pr[X \ge 0.8] = 1 - \frac{1}{1+e^{-0.8}} = \frac{1}{1+e^{0.8}} \approx 0.3100$$

$$\Pr[X \le 0.3] = \frac{1}{1+e^{-0.3}} \approx 0.5744$$

### c. $Y = 2X - 1$

With $a = -1$, $b = 2$:

$$F_Y(y) = F_X\!\left(\frac{y+1}{2}\right) = \frac{1}{1 + e^{-(y+1)/2}}$$

Density $f_Y(y) = \frac{1}{2}F_Y(y)(1-F_Y(y))$. Same S-shape, now centered at $y = -1$ and twice as wide (flatter slope) — see the right panel.

$$\Pr[Y < 0] = \frac{1}{1+e^{-1/2}} \approx 0.6225$$

Equivalently $\Pr[Y<0] = \Pr[X < 1/2] = F_X(0.5) \approx 0.6225$. ✓

---

## Q5 — Medians and quantile functions

### a. Median of the exponential

Solve $F(m) = 1 - e^{-m} = \tfrac12$:

$$e^{-m} = \tfrac12 \;\Rightarrow\; m = \ln 2 \approx 0.693$$

Note the median ($0.693$) is below the mean ($1$) — the right skew pulls the mean up.

### b. Median of the logistic

Solve $\dfrac{1}{1+e^{-m}} = \tfrac12 \Rightarrow 1 + e^{-m} = 2 \Rightarrow e^{-m} = 1$:

$$m = 0$$

Symmetric, so median = mean = mode = 0.

### c. Quantile function of the exponential

Set $u = 1 - e^{-x}$ and solve for $x$:

$$e^{-x} = 1-u \;\Rightarrow\; F^{-1}(u) = -\ln(1-u), \qquad u \in [0,1)$$

Check: $F^{-1}(0.5) = -\ln(0.5) = \ln 2$ ✓ (matches part a).

### d. Quantile function of the logistic

Set $u = \dfrac{1}{1+e^{-x}}$:

$$1 + e^{-x} = \frac{1}{u} \;\Rightarrow\; e^{-x} = \frac{1-u}{u} \;\Rightarrow\; F^{-1}(u) = \ln\!\left(\frac{u}{1-u}\right)$$

This is the **logit** function — the inverse of the sigmoid, and the link function in logistic regression. Check: $F^{-1}(0.5) = \ln 1 = 0$ ✓.

*(Both of these are what you'd use for inverse-transform sampling: draw $U \sim \text{Uniform}(0,1)$, then $F^{-1}(U)$ has the target distribution.)*

---

## Q6 — METABRIC

`n = 1343` patients; 286 received chemotherapy (21.3%), 1057 did not.

![Q6 ECDFs](solutions/q6_metabric_ecdf.png)

### a. ECDF of Overall Survival (Months)

Left panel. Survival ranges from 0.1 to 351 months with a median of about **118.5 months** (~9.9 years) and a mean of 129.1. The curve rises roughly linearly through the middle and flattens in the long right tail — a sizable minority of patients are followed for 250+ months.

### b. ECDF hued by Chemotherapy — what should we predict?

**Shorter.** Conditional on a patient having received chemotherapy, we should predict a *shorter* survival time.

The chemotherapy ECDF (red) lies entirely **above and to the left** of the no-chemotherapy ECDF (blue). At every survival time $t$, a larger fraction of the chemo group has already died/been censored by $t$:

| | median | 25th pct | 75th pct | $\Pr[T > 100\text{ mo}]$ | $\Pr[T > 200\text{ mo}]$ |
|---|---|---|---|---|---|
| No chemo (n=1057)  | 125.3 mo | 70.5 | 198.3 | 0.62 | 0.24 |
| Chemo (n=286)      | 90.3 mo  | 39.4 | 146.4 | 0.44 | 0.13 |

The chemo group's median is ~35 months lower, and the gap holds across the whole distribution — this is (first-order) stochastic dominance, not an artifact of one summary statistic. So as a purely *predictive* statement, `Chemotherapy = YES` is a marker for worse expected survival in this dataset.

### c. Is chemotherapy an effective treatment?

**This data cannot answer that question, and the negative association above is not evidence that chemo is harmful.** Part b is a statement about *prediction*; effectiveness is a statement about *causation*, and these are not the same thing.

This is observational data, not a randomized trial. Chemotherapy was **assigned by oncologists to the patients with worse disease** — that's confounding by indication. Confirming it in the data:

| median | No chemo | Chemo |
|---|---|---|
| Age at diagnosis | 64.0 | 50.2 |
| Tumor size (mm) | 21.0 | 25.0 |
| Positive lymph nodes | 0 | 2 |
| Nottingham prognostic index | 4.03 | 5.05 |

Chemo patients have larger tumors, nodal involvement, and a worse NPI (higher = worse prognosis) — all strong independent predictors of death. So the comparison in part b is contrasting *sicker patients who got treated* against *healthier patients who didn't*. The treatment effect and the prognostic effect are hopelessly entangled: what we observe is

$$\underbrace{E[T \mid \text{chemo}] - E[T \mid \text{no chemo}]}_{\text{what the ECDFs show}} = \underbrace{\text{causal effect}}_{\text{probably positive}} + \underbrace{\text{selection bias}}_{\text{strongly negative}}$$

and the second term is clearly dominating. (Note also that chemo patients are ~14 years younger, which cuts the *other* way and makes the observed deficit even more likely to be driven by disease severity.)

There's a second problem: `Overall Survival (Months)` mixes deaths with censoring — 41% of the no-chemo group and 48% of the chemo group are still living, so their recorded times are follow-up durations, not survival times. Treating that column as if it were a clean outcome biases any comparison, and the direction depends on enrollment timing. Proper analysis needs Kaplan–Meier / Cox methods that handle censoring explicitly.

**What would actually answer the question:** a randomized controlled trial, or — failing that — adjustment for the measured confounders (stratify or match on stage, tumor size, nodal status, NPI, and age; or fit a Cox model with those covariates) and a censoring-aware estimator. Randomized trials have in fact established that adjuvant chemotherapy improves breast cancer survival; the raw ECDF comparison here just reflects who gets prescribed it.
