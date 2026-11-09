import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

from mlxtend.frequent_patterns import apriori, association_rules

# krijojme folderin ku ruhen rezultatet
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
DATASET_PATH = "data/imdb-movies-dataset.csv"
# ngarkimi i dataset-it
df = pd.read_csv(DATASET_PATH)

cols = [
    "Year","Certificate","Duration (min)","Genre","Rating",
    "Metascore","Director","Votes","Review Count"
]

df = df[cols].copy()

# pasqyre fillestare e te dhenave
print("\n- DATA SUMMARY -")

# total columns
print("Total columns:", df.shape[1])

# numeric vs categorical
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

print("\nNumeric column names:", numeric_cols)
print("Categorical column names:", categorical_cols)

# kontrollojme vlerat qe mungojne
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\nMissing values per column:")
print(missing if len(missing) > 0 else "No missing values")

# pastrim i kolonave numerike qe kane presje si string
df["Votes"] = pd.to_numeric(df["Votes"].astype(str).str.replace(",",""), errors="coerce")
df["Review Count"] = pd.to_numeric(df["Review Count"].astype(str).str.replace(",",""), errors="coerce")

# marrim vetem zhanrin kryesor
df["Genre"] = df["Genre"].astype(str).str.split(",").str[0].str.strip()

# krijimi i kategorive per rating
def rating_cat(r):
    if pd.isna(r): return np.nan
    if r < 5.5: return "Bad"
    if r <= 7.0: return "Average"
    return "Good"

df["RatingCategory"] = df["Rating"].apply(rating_cat)
# heqim rreshtat pa target
df = df.dropna(subset=["RatingCategory"])

num_cols = ["Year","Duration (min)","Metascore","Votes","Review Count"]
cat_cols = ["Certificate","Genre","Director"]


# --- BEFORE CLUSTERING ---

df_before = df.copy()

cluster_cols = ["Metascore", "Votes", "Review Count"]

# mbushje missing values me mediane
for c in cluster_cols:
    df_before[c] = df_before[c].fillna(df_before[c].median())

X_before = df_before[cluster_cols]

# aplikimi i K-Means pa shume perpunim
kmeans_b = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
clusters_b = kmeans_b.fit_predict(X_before)

# metrika per vleresim te klastereve
sil_b = silhouette_score(X_before, clusters_b)
db_b = davies_bouldin_score(X_before, clusters_b)

# vizualizim me PCA ne 2D
pca_b = PCA(2)
Xb = pca_b.fit_transform(X_before)

plt.figure()
plt.scatter(Xb[:, 0], Xb[:, 1], c=clusters_b)
plt.title("BEFORE CLUSTERING")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_before.png"))
plt.close()

print("\n- BEFORE CLUSTERING -")
print("Columns used:", cluster_cols)
print("Silhouette:", sil_b)
print("Davies-Bouldin:", db_b)
print("Cluster sizes:")
print(pd.Series(clusters_b).value_counts())


# --- AFTER CLUSTERING ---

df_after = df.copy()

cluster_cols = ["Metascore", "Votes", "Review Count"]

# mbushje missing values per kolonat kryesore
for c in ["Year", "Duration (min)"]:
    df_after[c] = df_after[c].fillna(df_after[c].median())

for c in ["Genre", "Certificate", "Director"]:
    df_after[c] = df_after[c].fillna("Unknown")

# kufizim i regjisoreve (top 20 + Other)
top_directors = df_after["Director"].value_counts().head(20).index
df_after["Director"] = df_after["Director"].apply(
    lambda x: x if x in top_directors else "Other"
)

# trajtimi i outliers (prerje e vlerave ekstreme)
for c in cluster_cols:
    low, high = df_after[c].quantile([0.05, 0.95])
    df_after[c] = df_after[c].clip(low, high)

for c in cluster_cols:
    df_after[c] = df_after[c].fillna(df_after[c].median())

# transformim log per shperndarje me stabile
df_after["Votes"] = np.log1p(df_after["Votes"])
df_after["Review Count"] = np.log1p(df_after["Review Count"])

X_after = df_after[cluster_cols].copy()

# normalizim robust (i qendrueshem ndaj outliers)
scaler = RobustScaler()
X_after_scaled = scaler.fit_transform(X_after)

# KMeans pas perpunimit
kmeans_a = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20)
clusters_a = kmeans_a.fit_predict(X_after_scaled)

# metrika pas perpunimit
sil_a = silhouette_score(X_after_scaled, clusters_a)
db_a = davies_bouldin_score(X_after_scaled, clusters_a)

# vizualizim me PCA
pca_a = PCA(2)
Xa = pca_a.fit_transform(X_after_scaled)

plt.figure()
plt.scatter(Xa[:, 0], Xa[:, 1], c=clusters_a)
plt.title("AFTER CLUSTERING")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_after.png"))
plt.close()

print("\n- AFTER CLUSTERING -")
print("Columns used:", cluster_cols)
print("Silhouette:", sil_a)
print("Davies-Bouldin:", db_a)
print("Cluster sizes:")
print(pd.Series(clusters_a).value_counts())

df_after["Cluster"] = clusters_a
print("\nCluster means AFTER:")
print(df_after.groupby("Cluster")[cluster_cols].mean())


# --- APRIORI - RREGULLAT E SHOQËRIMIT ---

df_rules = df_after.copy()

# kthejme numrat ne kategori per analize me te lehte
df_rules["VotesCat"] = pd.cut(df_rules["Votes"], 3, labels=["Low","Medium","High"])
df_rules["MetaCat"] = pd.cut(df_rules["Metascore"], 3, labels=["Low","Medium","High"])
df_rules["DurCat"] = pd.cut(df_rules["Duration (min)"], 3, labels=["Short","Medium","Long"])
df_rules["YearCat"] = pd.cut(df_rules["Year"], bins=[0, 2000, 2015, 3000], labels=["Before_2000", "2000_2015", "2015_plus"]
)

rules_df = df_rules[
    ["Genre","Certificate","Director","RatingCategory","VotesCat","MetaCat","DurCat", "YearCat"]
]

# encoding per algoritmin apriori
encoded = pd.get_dummies(rules_df)

# nxjerrja e frekuencave
freq = apriori(encoded, min_support=0.05, use_colnames=True)

# krijimi i rregullave
rules = association_rules(freq, metric="confidence", min_threshold=0.6)
rules = rules.sort_values("lift", ascending=False)

# thjeshtim i struktures se rregullave
def clean_set(x):
    return ", ".join(list(x))

rules["antecedents"] = rules["antecedents"].apply(clean_set)
rules["consequents"] = rules["consequents"].apply(clean_set)

# mbajme vetem metrikat kryesore
rules = rules[[
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift"
]]


# heqim rregulla jo te vlefshme ose te zhurmshme
def is_noisy(rule):
    a = str(rule["antecedents"])
    c = str(rule["consequents"])

    if "Unknown" in a or "Unknown" in c:
        return True

    if "RatingCategory_Good" in a and "MetaCat_High" in c:
        return True

    return False


rules_clean = rules[~rules.apply(is_noisy, axis=1)]

# filtrim per rregulla interesante
interesting = rules_clean[
    (rules_clean["lift"] >= 1.2) &
    (rules_clean["confidence"] >= 0.55) &
    (rules_clean["support"] >= 0.03)
]

# kerkojme rregulla me komplekse (jo triviale)
interesting = interesting[
    interesting["antecedents"].apply(lambda x: len(str(x).split(",")) >= 2)]


# fokus vetem ne rregulla me “kuptimplote” 
interesting_story = interesting[
    (
        interesting["antecedents"].str.contains("DurCat") |
        interesting["antecedents"].str.contains("Certificate") |
        interesting["antecedents"].str.contains("Director") |
        interesting["antecedents"].str.contains("Genre") |
        interesting["antecedents"].str.contains("YearCat")
    )
]


# print final i rezultateve
print("\n- TOP INTERESTING ASSOCIATION RULES -")

print(
    interesting_story[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]]
    .sort_values(by="lift", ascending=False)
    .head(20)
)

# ruajtja e rezultateve
rules_clean.to_csv(os.path.join(OUTPUT_DIR, "all_rules_clean.csv"), index=False)
interesting.to_csv(os.path.join(OUTPUT_DIR, "interesting_rules.csv"), index=False)