import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Leximi i datasetit
df = pd.read_csv("data/imdb-movies-dataset.csv")

# Shfaqim formen e datasetit dhe emrat e kolonave
print("Forma e datasetit:", df.shape)
print("Kolonat:")
print(df.columns.tolist())

# EDA - Exploratory Data Analysis
# Kjo pjese na ndihmon me kuptu datasetin para se me fillu punen
print("\n===== EDA =====")

# Info na tregon tipin e kolonave dhe sa vlera jo-null ka secila
print("\nInfo:")
df.info()

# Describe jep statistika baze per kolonat
print("\nDescribe:")
print(df.describe(include="all"))

# Kontrollojme sa missing values ka secila kolone
print("\nMissing values per column:")
print(df.isnull().sum())

# Zgjedhim vetem kolonat qe na duhen per analize
df = df[[
    "Year",
    "Certificate",
    "Duration (min)",
    "Genre",
    "Rating",
    "Metascore",
    "Director",
    "Votes",
    "Review Count"
]].copy()

# EDA pas zgjedhjes se kolonave
print("\n===== EDA PAS ZGJEDHJES SE KOLONAVE =====")
print("\nInfo:")
df.info()

print("\nDescribe:")
print(df.describe(include="all"))

print("\nMissing values per selected column:")
print(df.isnull().sum())

# Shikojme sa vlera unike kane kolonat kategorike
print("\nNumri i vlerave unike per kolonat kategorike:")
for col in ["Certificate", "Genre", "Director"]:
    print(f"{col}: {df[col].nunique()} unique values")

# Pastrimi i kolonave numerike
# Te Votes dhe Review Count mund te kete presje, prandaj i kthejme ne numra
df["Votes"] = pd.to_numeric(
    df["Votes"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

df["Review Count"] = pd.to_numeric(
    df["Review Count"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

# Feature engineering
# Te Genre marrim vetem zhanrin e pare
# Kjo behet qe te mos kemi shume kategori dhe te fryhet dataseti
df["Genre"] = df["Genre"].astype(str).str.split(",").str[0].str.strip()

# Krijimi i target variable
# RatingCategory do te jete variabla qe duam ta parashikojme
# 1 = low, 2 = medium, 3 = high
def categorize_rating(r):
    if pd.isna(r):
        return np.nan
    elif r < 5.5:
        return 1
    elif r <= 7.0:
        return 2    
    else:
        return 3

# Aplikimi i funksionit ne kolonen Rating
df["RatingCategory"] = df["Rating"].apply(categorize_rating)

# Heqim rreshtat ku target variabla eshte missing
df = df.dropna(subset=["RatingCategory"]).copy()
df["RatingCategory"] = df["RatingCategory"].astype(int)

# Shfaqim sa filma ka ne secilen klase
print("\nShperndarja e klasave te target-it:")
print(df["RatingCategory"].value_counts().sort_index())

# Percaktojme kolonat numerike dhe kategorike
num_cols = ["Year", "Duration (min)", "Metascore", "Votes", "Review Count"]
cat_cols = ["Certificate", "Genre", "Director"]

# BEFORE PREPROCESSING
# Kjo pjese paraqet nje model baze me trajtim minimal te te dhenave
df_before = df.copy()

# Plotesimi i missing values per kolonat numerike me median
for col in num_cols:
    df_before[col] = df_before[col].fillna(df_before[col].median())

# Plotesimi i missing values per kolonat kategorike me "Unknown"
for col in cat_cols:
    df_before[col] = df_before[col].fillna("Unknown")

# One-hot encoding per kolonat kategorike
df_before = pd.get_dummies(df_before, columns=cat_cols, drop_first=False)

# Ndarja ne X dhe y
# X eshte test, y eshte target variabla
Xb = df_before.drop(columns=["Rating", "RatingCategory"])
yb = df_before["RatingCategory"]

# Ndarja e te dhenave ne train dhe test
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.2, random_state=42, stratify=yb
)

# Krijimi dhe trajnimi i Logistic Regression
lr_before = LogisticRegression(max_iter=2000, random_state=42)
lr_before.fit(Xb_train, yb_train)
yb_pred_lr = lr_before.predict(Xb_test)

# Krijimi dhe trajnimi i Random Forest
rf_before = RandomForestClassifier(n_estimators=200, random_state=42)
rf_before.fit(Xb_train, yb_train)
yb_pred_rf = rf_before.predict(Xb_test)

# Llogaritja e accuracy per te dy modelet
acc_lr_before = accuracy_score(yb_test, yb_pred_lr)
acc_rf_before = accuracy_score(yb_test, yb_pred_rf)

# Rezultatet para preprocessing
print("\n===== BEFORE PREPROCESSING =====")
print("Logistic Regression Accuracy:", acc_lr_before)
print("Random Forest Accuracy:", acc_rf_before)

print("\nClassification Report - Logistic Regression (Before):")
print(classification_report(yb_test, yb_pred_lr))

print("\nConfusion Matrix - Logistic Regression (Before):")
print(confusion_matrix(yb_test, yb_pred_lr))

print("\nClassification Report - Random Forest (Before):")
print(classification_report(yb_test, yb_pred_rf))

print("\nConfusion Matrix - Random Forest (Before):")
print(confusion_matrix(yb_test, yb_pred_rf))

# AFTER PREPROCESSING
# Kjo pjese paraqet versionin e perpunuar te datasetit
df_after = df.copy()

# Plotesimi i missing values per kolonat kategorike
df_after["Certificate"] = df_after["Certificate"].fillna("Unknown")
df_after["Genre"] = df_after["Genre"].fillna("Unknown")
df_after["Director"] = df_after["Director"].fillna("Unknown")

# Te Director ka shume emra te ndryshem
# Per kete arsye mbajme vetem 20 regjisoret me te shpeshte
# Te tjeret i vendosim si "Other"
top_directors = df_after["Director"].value_counts().head(20).index
df_after["Director"] = df_after["Director"].apply(
    lambda x: x if x in top_directors else "Other"
)

# Plotesimi i missing values per kolonat numerike me median
for col in num_cols:
    df_after[col] = df_after[col].fillna(df_after[col].median())

# Trajtimi i outliers me quantile clipping
# Kufizojme vlerat ekstreme mes 5% dhe 95%
for col in num_cols:
    low = df_after[col].quantile(0.05)
    high = df_after[col].quantile(0.95)
    df_after[col] = df_after[col].clip(low, high)

# One-hot encoding per kolonat kategorike
df_after = pd.get_dummies(df_after, columns=cat_cols, drop_first=False)

# Ndarja ne X dhe y
Xa = df_after.drop(columns=["Rating", "RatingCategory"])
ya = df_after["RatingCategory"]

# Ndarja ne train dhe test
Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    Xa, ya, test_size=0.2, random_state=42, stratify=ya
)

# Standardizimi behet vetem per kolonat numerike
# Kjo eshte me e rendesishme sidomos per Logistic Regression
scaler = StandardScaler()
Xa_train[num_cols] = scaler.fit_transform(Xa_train[num_cols])
Xa_test[num_cols] = scaler.transform(Xa_test[num_cols])

# Logistic Regression pas preprocessing
lr_after = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
lr_after.fit(Xa_train, ya_train)
ya_pred_lr = lr_after.predict(Xa_test)

# Random Forest pas preprocessing
rf_after = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)
rf_after.fit(Xa_train, ya_train)
ya_pred_rf = rf_after.predict(Xa_test)

# Llogaritja e accuracy
acc_lr_after = accuracy_score(ya_test, ya_pred_lr)
acc_rf_after = accuracy_score(ya_test, ya_pred_rf)

# Rezultatet pas preprocessing
print("\n===== AFTER PREPROCESSING =====")
print("Logistic Regression Accuracy:", acc_lr_after)
print("Random Forest Accuracy:", acc_rf_after)

print("\nClassification Report - Logistic Regression (After):")
print(classification_report(ya_test, ya_pred_lr))

print("\nConfusion Matrix - Logistic Regression (After):")
print(confusion_matrix(ya_test, ya_pred_lr))

print("\nClassification Report - Random Forest (After):")
print(classification_report(ya_test, ya_pred_rf))

print("\nConfusion Matrix - Random Forest (After):")
print(confusion_matrix(ya_test, ya_pred_rf))

# Krahasimi final i rezultateve
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Before Preprocessing": [acc_lr_before, acc_rf_before],
    "After Preprocessing": [acc_lr_after, acc_rf_after]
})

print("\n===== FINAL COMPARISON =====")
print(comparison)

# Testimi i modelit me nje film te ri
# Kjo pjese perdoret per me provu modelin me te dhena te reja
def predict_movie(movie_dict):
    df_new = pd.DataFrame([movie_dict])

    # Plotesimi i missing values per kolonat kategorike
    df_new["Certificate"] = df_new["Certificate"].fillna("Unknown")
    df_new["Genre"] = df_new["Genre"].fillna("Unknown")
    df_new["Director"] = df_new["Director"].fillna("Unknown")

    # Nese regjisori nuk eshte ne listen e top_directors,
    # atehere e vendosim si "Other"
    if df_new["Director"].iloc[0] not in top_directors:
        df_new["Director"] = "Other"

    # Plotesimi i missing values per kolonat numerike
    for col in num_cols:
        df_new[col] = df_new[col].fillna(df_after[col].median())

    # Kufizimi i outliers me te njejten logjike si te preprocessing
    for col in num_cols:
        low = df[col].fillna(df[col].median()).quantile(0.05)
        high = df[col].fillna(df[col].median()).quantile(0.95)
        df_new[col] = df_new[col].clip(low, high)

    # One-hot encoding per te dhenat e reja
    df_new = pd.get_dummies(df_new)

    # Pershtatja e kolonave me ato qe ka pasur dataseti i trajnimit
    df_new = df_new.reindex(columns=Xa.columns, fill_value=0)

    # Standardizimi i kolonave numerike
    df_new[num_cols] = scaler.transform(df_new[num_cols])

    # Parashikimi me Random Forest
    return rf_after.predict(df_new)[0]

# Shembull testimi me nje film te ri
print("\nTest prediction for one movie:")
print(
    predict_movie({
        "Year": 2020,
        "Duration (min)": 150,
        "Metascore": 85,
        "Votes": 800000,
        "Review Count": 5000,
        "Certificate": "PG-13",
        "Director": "Christopher Nolan",
        "Genre": "Action"
    })
)