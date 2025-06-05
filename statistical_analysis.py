import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr, kruskal, mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

# Load and clean data
data = pd.read_csv('/content/celiac_disease_lab_data.csv')
df = data.copy()

df['Marsh'] = df['Marsh'].astype(str).str.lower().str.strip()
df['Marsh'] = df['Marsh'].replace({
    'marsh type 0': '0',
    'marsh type 1': '1',
    'marsh type 2': '2',
    'marsh type 3a': '3a',
    'marsh type 3b': '3b',
    'marsh type 3c': '3c',
    'none': np.nan,
    'nan': np.nan,
})

marsh_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3a': 3,
    '3b': 4,
    '3c': 5,
}

df['Marsh'] = df['Marsh'].map(marsh_mapping)
df = df.dropna(subset=['Marsh'])
df['Marsh'] = df['Marsh'].astype(int)

# Clean feature columns
feature_cols = [
    'Diabetes Type', 'Short_Stature', 'Sticky_Stool', 'Weight_loss',
    'IgA', 'IgG', 'IgM', 'Age', 'Gender', 'Diabetes', 'Diarrhoea', 'Abdominal'
]

for col in feature_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()
    df[col] = df[col].replace({'none': np.nan, 'nan': np.nan})

df = df.dropna(subset=feature_cols)

# Label encode categorical columns
cat_cols = ['Gender', 'Diabetes', 'Diabetes Type', 'Diarrhoea', 'Abdominal',
            'Short_Stature', 'Sticky_Stool', 'Weight_loss']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Convert numeric features
num_cols = ['IgA', 'IgG', 'IgM', 'Age']
for col in num_cols:
    df[col] = df[col].astype(float)

# === Spearman correlation ===
print("\nSpearman Correlation with Marsh:")
for col in ['IgA', 'IgG', 'Age']:
    r, p = spearmanr(df[col], df['Marsh'])
    print(f"{col}: r = {r:.3f}, p = {p:.3e}")

#  Kruskal-Wallis H Test 
print("\nKruskal-Wallis Tests:")
for col in ['IgA', 'IgG', 'Age']:
    groups = [df[df['Marsh'] == m][col] for m in sorted(df['Marsh'].unique())]
    stat, p = kruskal(*groups)
    print(f"{col}: H = {stat:.3f}, p = {p:.3e}")

#  Mann-Whitney U Test (0 vs others) 
print("\nPairwise Mann-Whitney (0 vs others):")
for m in sorted(df['Marsh'].unique()):
    if m == 0:
        continue
    for col in ['IgA', 'IgG', 'Age']:
        u, p = mannwhitneyu(df[df['Marsh'] == 0][col], df[df['Marsh'] == m][col])
        print(f"Marsh 0 vs {m} - {col}: U = {u:.1f}, p = {p:.3e}")

# All Pairwise Mann-Whitney U Tests with Bonferroni (IgM) 
print("\nPairwise Mann-Whitney (IgM) with Bonferroni correction:")
pairs = list(itertools.combinations(sorted(df['Marsh'].unique()), 2))
p_vals = []

for a, b in pairs:
    u, p = mannwhitneyu(df[df['Marsh'] == a]['IgM'], df[df['Marsh'] == b]['IgM'])
    p_vals.append(p)

adj_p = multipletests(p_vals, method='bonferroni')[1]
for (a, b), raw_p, bonf_p in zip(pairs, p_vals, adj_p):
    print(f"IgM: Marsh {a} vs {b} - raw p = {raw_p:.3e}, adjusted p = {bonf_p:.3e}")

#  Pairwise Chi-Square (Abdominal) 
print("\nChi-Square Tests (Abdominal vs Marsh):")
chi_pvals = []
results = []

for a, b in pairs:
    subset = df[df['Marsh'].isin([a, b])]
    table = pd.crosstab(subset['Marsh'], subset['Abdominal'])
    chi2, p, _, _ = chi2_contingency(table)
    results.append((a, b, chi2, p))
    chi_pvals.append(p)

adj_chi_p = multipletests(chi_pvals, method='bonferroni')[1]
for (a, b, chi2, raw_p), adj in zip(results, adj_chi_p):
    print(f"Abdominal: Marsh {a} vs {b} - Chi2 = {chi2:.2f}, raw p = {raw_p:.3e}, adj p = {adj:.3e}")

# Visualization 

# Boxplot: IgM across Marsh
plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=df, x='Marsh', y='IgM')
pairs_to_annotate = [((0, 1), 0.02), ((0, 2), 0.001)]

annotator = Annotator(ax, pairs=[p[0] for p in pairs_to_annotate], data=df, x='Marsh', y='IgM')
annotator.configure(test=None, text_format='star', loc='inside')
annotator.set_pvalues([p[1] for p in pairs_to_annotate])
annotator.annotate()
plt.title("IgM Across Marsh Levels")
plt.show()

# Countplot: Abdominal vs Marsh
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Marsh', hue='Abdominal')
plt.title("Abdominal Pain Across Marsh Levels")
plt.show()



