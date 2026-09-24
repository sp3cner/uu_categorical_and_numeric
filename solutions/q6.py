import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('./data/metabric.csv')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

# a.
sns.ecdfplot(data=df, x='Overall Survival (Months)', ax=axes[0])
axes[0].set_title('a. ECDF of overall survival')

# b.
sns.ecdfplot(data=df, x='Overall Survival (Months)', hue='Chemotherapy', ax=axes[1])
axes[1].set_title('b. ECDF by chemotherapy')

for ax in axes:
    ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig('solutions/q6_ecdf_seaborn.png', dpi=150)

# ---- numbers for the write-up
g = df.groupby('Chemotherapy')['Overall Survival (Months)']
print(g.agg(n='count', mean='mean', median='median',
            q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75)).round(1))
print("\nfraction still alive at t months:")
print(pd.DataFrame({t: g.apply(lambda s, t=t: (s > t).mean())
                    for t in (50, 100, 150, 200)}).round(3))
print("\nECDF height at t  (fraction at or below t):")
print(pd.DataFrame({t: g.apply(lambda s, t=t: (s <= t).mean())
                    for t in (50, 100, 150, 200)}).round(3))
print("\nmedian covariates by group:")
print(df.groupby('Chemotherapy')[['Age at Diagnosis', 'Tumor Size',
      'Lymph nodes examined positive', 'Nottingham prognostic index',
      'Tumor Stage']].median().round(2))
print("\ncensoring (survival status):")
print(pd.crosstab(df['Chemotherapy'], df['Overall Survival Status'], normalize='index').round(3))
