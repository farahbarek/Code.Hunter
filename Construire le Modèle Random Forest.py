
"""
Construction du modèle Random Forest pour prédiction de solvabilité d'homicides.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================
print("=" * 70)
print("CONSTRUCTION DU MODÈLE RANDOM FOREST")
print("=" * 70)

chemin_fichier = r"C:\homicide_data_mining\database.csv"
df = pd.read_csv(chemin_fichier)
print(f"\n✅ {len(df):,} cas chargés")

# Préparation variable cible
df['Solved'] = df['Crime Solved'].map({'Yes': 1, 'No': 0})
print(f"   Résolus : {df['Solved'].sum():,} ({df['Solved'].mean():.1%})")
print(f"   Non-résolus : {(df['Solved']==0).sum():,}")

# ============================================================================
# PRÉPARATION DES DONNÉES
# ============================================================================
print("\n" + "=" * 70)
print("PRÉPARATION")
print("=" * 70)

# Catégorisation âge
def categorize_age(age):
    if pd.isna(age): return 'Unknown'
    try:
        a = float(age)
        if a < 18: return 'Child'
        elif a < 30: return 'Young Adult'
        elif a < 50: return 'Adult'
        elif a < 65: return 'Middle Aged'
        else: return 'Senior'
    except: return 'Unknown'

# Groupe d'armes
def weapon_group(w):
    if pd.isna(w): return 'Unknown'
    w_lower = str(w).lower()
    if any(k in w_lower for k in ['firearm', 'gun', 'handgun', 'shotgun', 'rifle']):
        return 'Firearm'
    elif any(k in w_lower for k in ['knife', 'cut', 'stab']):
        return 'Knife'
    elif 'blunt' in w_lower:
        return 'Blunt Object'
    elif 'strangulation' in w_lower:
        return 'Strangulation'
    else:
        return 'Other'

# Application des transformations
if 'Victim Age' in df.columns:
    df['Victim_Age_Category'] = df['Victim Age'].apply(categorize_age)

if 'Weapon' in df.columns:
    df['Weapon_Group'] = df['Weapon'].apply(weapon_group)

if 'State' in df.columns:
    state_rates = df.groupby('State')['Solved'].mean().to_dict()
    df['State_Historical_Solve_Rate'] = df['State'].map(state_rates)

# Sélection des features
key_features = [
    'Victim Sex', 'Victim Age', 'Victim Race',
    'Victim_Age_Category', 'Weapon_Group',
    'Victim Count', 'Year', 'State',
    'State_Historical_Solve_Rate'
]

available_features = [f for f in key_features if f in df.columns]
print(f"\n✅ {len(available_features)} features sélectionnées")

X = df[available_features].copy()
y = df['Solved'].copy()

# Encodage intelligent - limiter les catégories
for col in X.select_dtypes(include=['object']).columns:
    top_cats = X[col].value_counts().head(15).index
    X[col] = X[col].where(X[col].isin(top_cats), 'Other')

X_encoded = pd.get_dummies(X, drop_first=True)
print(f"✅ {X_encoded.shape[1]} features après encodage")

# ============================================================================
# DIVISION DES DONNÉES
# ============================================================================
print("\n" + "=" * 70)
print("DIVISION TRAIN/TEST")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"\n✅ Train : {X_train.shape[0]:,} ({y_train.mean():.1%})")
print(f"✅ Test : {X_test.shape[0]:,} ({y_test.mean():.1%})")

# ============================================================================
# ENTRAÎNEMENT DU MODÈLE
# ============================================================================
print("\n" + "=" * 70)
print("ENTRAÎNEMENT RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

print("\n🌳 Entraînement...")
model.fit(X_train, y_train)
print("✅ Modèle entraîné")

# ============================================================================
# ÉVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("ÉVALUATION")
print("=" * 70)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n📊 AUC-ROC : {auc:.4f}")

print("\n📋 Rapport :")
print(classification_report(y_test, y_pred, target_names=['Non-Résolu', 'Résolu']))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"\n✔️  Précision : {accuracy:.2%}")

# ============================================================================
# IMPORTANCE DES FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("TOP 15 FEATURES")
print("=" * 70)

feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i in range(min(15, len(feature_importance))):
    feat = feature_importance.iloc[i]
    print(f"{i+1:2}. {feat['feature'][:45]:45} : {feat['importance']:.4f}")

# Visualisation
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=9)
plt.xlabel('Importance')
plt.title('Top 20 Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
plt.show()

# ============================================================================
# ANALYSE CAS NON-RÉSOLUS
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSE CAS NON-RÉSOLUS")
print("=" * 70)

unsolved_mask = df['Solved'] == 0
unsolved_count = unsolved_mask.sum()
print(f"\n📌 Cas non-résolus : {unsolved_count:,}")

if unsolved_count > 0:
    unsolved_features = X_encoded[unsolved_mask]
    unsolved_probs = model.predict_proba(unsolved_features)[:, 1]

    print(f"\n📈 Statistiques :")
    print(f"   Moyenne : {unsolved_probs.mean():.3f}")
    print(f"   Médiane : {np.median(unsolved_probs):.3f}")
    print(f"   Min-Max : {unsolved_probs.min():.3f} - {unsolved_probs.max():.3f}")

    # Catégorisation
    print(f"\n🎯 Distribution :")
    categories = [
        (0.8, 1.0, "Très élevée (>80%)"),
        (0.6, 0.8, "Élevée (60-80%)"),
        (0.4, 0.6, "Modérée (40-60%)"),
        (0.2, 0.4, "Faible (20-40%)"),
        (0.0, 0.2, "Très faible (<20%)")
    ]

    for low, high, label in categories:
        if high == 1.0:
            count = (unsolved_probs > low).sum()
        else:
            count = ((unsolved_probs >= low) & (unsolved_probs < high)).sum()
        pct = count / len(unsolved_probs) * 100
        print(f"   {label} : {count:,} cas ({pct:.1f}%)")

    # Sauvegarder résultats
    results_df = df[unsolved_mask].copy()
    results_df['Predicted_Solvability'] = unsolved_probs
    results_df = results_df.sort_values('Predicted_Solvability', ascending=False)

    results_df.to_csv('cas_non_resolus_classifies.csv', index=False)
    print(f"\n💾 Résultats sauvegardés")

    # Top 5
    print(f"\n🏆 Top 5 :")
    for i in range(min(5, len(results_df))):
        case = results_df.iloc[i]
        print(f"\n{i+1}. Proba : {case['Predicted_Solvability']:.1%}")
        if 'Weapon' in case:
            print(f"   Arme : {case['Weapon']}")
        if 'Victim Sex' in case and 'Victim Age' in case:
            print(f"   Victime : {case['Victim Sex']}, {int(case['Victim Age'])} ans")

# ============================================================================
# SAUVEGARDE DU MODÈLE
# ============================================================================
print("\n" + "=" * 70)
print("SAUVEGARDE")
print("=" * 70)

joblib.dump(model, 'homicide_model_optimized.pkl')
with open('model_features_optimized.json', 'w') as f:
    json.dump(list(X_encoded.columns), f)

print("\n✅ Modèle sauvegardé : homicide_model_optimized.pkl")
print("✅ Features sauvegardées : model_features_optimized.json")

# Rapport final
print("\n" + "=" * 70)
print("RÉSUMÉ")
print("=" * 70)
print(f"""
✅ MODÈLE CONSTRUIT AVEC SUCCÈS

Performances :
• AUC-ROC : {auc:.4f}
• Accuracy : {accuracy:.2%}
• Train samples : {X_train.shape[0]:,}
• Test samples : {X_test.shape[0]:,}

Fichiers générés :
• homicide_model_optimized.pkl
• model_features_optimized.json
• cas_non_resolus_classifies.csv
• feature_importance.png
""")
print("=" * 70)

# Lecture du fichier CSV dans un DataFrame appelé 'df'
df = pd.read_csv(chemin_du_fichier)

# Affichage des 5 premières lignes du DataFrame pour confirmer que la lecture a réussi
print("Lecture réussie. Voici les premières lignes de votre DataFrame :")
print(df.head())
df.info()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# First, let's understand the target variable
print("Crime Solved value counts:")
print(df['Crime Solved'].value_counts())

# Convert to binary target (1 = solved, 0 = unsolved)
df['Solved'] = df['Crime Solved'].map({'Yes': 1, 'No': 0})

print(f"\nTotal cases: {len(df)}")
print(f"Solved cases: {df['Solved'].sum()} ({df['Solved'].mean():.2%})")
print(f"Unsolved cases: {len(df) - df['Solved'].sum()} ({1-df['Solved'].mean():.2%})")
# Split data as you suggested
solved_cases = df[df['Solved'] == 1]
unsolved_cases = df[df['Solved'] == 0]

def analyze_missing_patterns(df, solved_cases, unsolved_cases):
    print("=== MISSING DATA ANALYSIS ===")
    
    # Check for missing values in each group
    missing_solved = solved_cases.isnull().sum()
    missing_unsolved = unsolved_cases.isnull().sum()
    
    missing_analysis = pd.DataFrame({
        'Solved_Missing': missing_solved,
        'Unsolved_Missing': missing_unsolved,
        'Solved_Missing_Pct': (missing_solved / len(solved_cases)) * 100,
        'Unsolved_Missing_Pct': (missing_unsolved / len(unsolved_cases)) * 100
    })
    
    # Add difference in missing rates
    missing_analysis['Missing_Rate_Diff'] = (
        missing_analysis['Unsolved_Missing_Pct'] - missing_analysis['Solved_Missing_Pct']
    )
    
    return missing_analysis.sort_values('Missing_Rate_Diff', ascending=False)

missing_analysis = analyze_missing_patterns(df, solved_cases, unsolved_cases)
print(missing_analysis.head(20))
# Check data types and unique values for key columns
key_columns = ['Perpetrator Sex', 'Perpetrator Age', 'Perpetrator Race', 
               'Relationship', 'Weapon', 'Victim Sex', 'Victim Age', 'Victim Race']

for col in key_columns:
    print(f"\n--- {col} ---")
    print(f"Solved cases unique values: {solved_cases[col].unique()}")
    print(f"Unsolved cases unique values: {unsolved_cases[col].unique()}")

def preprocess_data(df):
    df_clean = df.copy()
    
    # Handle Perpetrator Age (might have unknown values)
    print("Perpetrator Age values:", df['Perpetrator Age'].unique())
    
    # Convert Perpetrator Age to numeric, coerce errors to NaN
    df_clean['Perpetrator Age'] = pd.to_numeric(df_clean['Perpetrator Age'], errors='coerce')
    
    # Create data quality flags
    df_clean['Perpetrator_Age_Missing'] = df_clean['Perpetrator Age'].isnull().astype(int)
    df_clean['Perpetrator_Sex_Unknown'] = (df_clean['Perpetrator Sex'] == 'Unknown').astype(int)
    df_clean['Weapon_Unknown'] = (df_clean['Weapon'] == 'Unknown').astype(int)
    df_clean['Relationship_Unknown'] = (df_clean['Relationship'] == 'Unknown').astype(int)
    
    # Fill missing ages with median
    df_clean['Perpetrator Age'].fillna(df_clean['Perpetrator Age'].median(), inplace=True)
    
    return df_clean

df_clean = preprocess_data(df)
# ============================================================================
# SUITE DU SCRIPT - À AJOUTER APRÈS VOTRE CODE EXISTANT
# ============================================================================

print("\n" + "="*60)
print("🚀 ÉTAPE SUIVANTE : PRÉPARATION POUR LE MODÈLE")
print("="*60)

# ------------------------------------------------------------
# ÉTAPE 1 : Sélection des features pour le modèle
# ------------------------------------------------------------
print("\n🔍 ÉTAPE 1 : Sélection des features...")

# Liste des features à utiliser (adaptée à vos colonnes)
features = [
    # Information sur la victime (toujours connue)
    'Victim Sex', 'Victim Age', 'Victim Race', 'Victim Ethnicity',
    
    # Information sur le suspect/circonstances
    'Perpetrator Sex', 'Perpetrator Age', 'Perpetrator Race', 
    'Perpetrator Ethnicity', 'Relationship', 'Weapon',
    
    # Comptes
    'Victim Count', 'Perpetrator Count',
    
    # Temporal/Géographique
    'Year', 'Month', 'State', 'City', 'Agency Type',
    
    # Flags de qualité de données (que vous avez créés)
    'Perpetrator_Age_Missing', 'Perpetrator_Sex_Unknown',
    'Weapon_Unknown', 'Relationship_Unknown'
]

# Vérifier quelles features existent réellement
available_features = []
for feature in features:
    if feature in df_clean.columns:
        available_features.append(feature)
    else:
        print(f"⚠️  Feature non trouvée: {feature}")

print(f"\n✅ {len(available_features)}/{len(features)} features disponibles")
print("Features utilisées:", available_features)

if len(available_features) < 10:
    print("\n❌ Problème: Pas assez de features!")
    print("Colonnes disponibles dans df_clean:")
    print(df_clean.columns.tolist())
    exit()

# ------------------------------------------------------------
# ÉTAPE 2 : Préparation de X et y
# ------------------------------------------------------------
print("\n🔧 ÉTAPE 2 : Préparation des données X et y...")

X = df_clean[available_features]
y = df_clean['Solved']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Proportion résolus dans y: {y.mean():.2%}")

# ------------------------------------------------------------
# ÉTAPE 3 : Encodage des variables catégorielles
# ------------------------------------------------------------
print("\n🔠 ÉTAPE 3 : Encodage des variables catégorielles...")

# Identifier les colonnes catégorielles
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Colonnes catégorielles ({len(categorical_cols)}): {categorical_cols}")
print(f"Colonnes numériques ({len(numerical_cols)}): {numerical_cols}")

# Encoder les variables catégorielles avec get_dummies
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"\n✅ Après encodage:")
print(f"   X_encoded shape: {X_encoded.shape}")
print(f"   Nombre total de features: {X_encoded.shape[1]}")

# Afficher quelques nouvelles colonnes créées
print("\nExemples de nouvelles colonnes créées:")
new_columns = X_encoded.columns.tolist()
print(f"Premières 10: {new_columns[:10]}")
print(f"Dernières 10: {new_columns[-10:]}")

# ------------------------------------------------------------
# ÉTAPE 4 : Division des données
# ------------------------------------------------------------
print("\n📊 ÉTAPE 4 : Division des données (Train/Test)...")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, 
    y, 
    test_size=0.3,        # 30% pour le test
    random_state=42,      # Pour la reproductibilité
    stratify=y           # Garde la même proportion de résolus/non-résolus
)

print(f"✅ Données divisées avec succès!")
print(f"   Ensemble d'entraînement: {X_train.shape[0]} cas")
print(f"   Ensemble de test: {X_test.shape[0]} cas")
print(f"   Proportion résolus dans train: {y_train.mean():.2%}")
print(f"   Proportion résolus dans test: {y_test.mean():.2%}")

# ------------------------------------------------------------
# ÉTAPE 5 : Construction du modèle Random Forest
# ------------------------------------------------------------
print("\n🌳 ÉTAPE 5 : Construction du Random Forest...")

from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(
    n_estimators=100,      # Nombre d'arbres
    max_depth=15,          # Profondeur maximale
    min_samples_split=20,  # Nombre minimum pour diviser un nœud
    min_samples_leaf=10,   # Nombre minimum dans une feuille
    random_state=42,       # Pour reproductibilité
    class_weight='balanced', # Gère le déséquilibre des classes
    n_jobs=-1              # Utilise tous les processeurs
)

print("Entraînement du modèle en cours...")
model.fit(X_train, y_train)
print("✅ Modèle entraîné avec succès!")

# ------------------------------------------------------------
# ÉTAPE 6 : Évaluation du modèle
# ------------------------------------------------------------
print("\n📈 ÉTAPE 6 : Évaluation du modèle...")

# Faire des prédictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilités d'être résolu

# 1. Rapport de classification
print("\n1️⃣  RAPPORT DE CLASSIFICATION:")
print("="*40)
print(classification_report(y_test, y_pred, target_names=['Non-Résolu', 'Résolu']))

# 2. Matrice de confusion
print("\n2️⃣  MATRICE DE CONFUSION:")
print("="*40)
cm = confusion_matrix(y_test, y_pred)

# Afficher la matrice de confusion de façon lisible
tn, fp, fn, tp = cm.ravel()
print(f"""
               Prédit Non-Résolu   Prédit Résolu
Réel Non-Résolu       {tn:>7}           {fp:>7}
Réel Résolu           {fn:>7}           {tp:>7}
""")

# Interprétation
print(f"\n📊 INTERPRÉTATION:")
print(f"✅ Vrais Positifs: {tp} (correctement prédits comme résolus)")
print(f"❌ Faux Positifs: {fp} (prédits résolus mais non-résolus)")
print(f"❌ Faux Négatifs: {fn} (prédits non-résolus mais résolus)")
print(f"✅ Vrais Négatifs: {tn} (correctement prédits comme non-résolus)")

# Calculer l'accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"\n📈 Précision globale: {accuracy:.2%}")

# ------------------------------------------------------------
# ÉTAPE 7 : Importance des features
# ------------------------------------------------------------
print("\n🔍 ÉTAPE 7 : Importance des features...")

# Créer un DataFrame avec l'importance des features
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 15 FEATURES LES PLUS IMPORTANTES:")
print("="*50)
for i in range(min(15, len(feature_importance))):
    feat = feature_importance.iloc[i]
    feature_name = feat['feature'][:50]  # Tronquer les noms longs
    print(f"{i+1:2}. {feature_name:50} : {feat['importance']:.4f}")

# Visualisation
plt.figure(figsize=(12, 8))
top_n = 20
top_features = feature_importance.head(top_n)

plt.barh(range(top_n), top_features['importance'])
plt.yticks(range(top_n), top_features['feature'])
plt.xlabel('Importance')
plt.title(f'Top {top_n} Features Influençant la Solvabilité')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# ÉTAPE 8 : Analyser les cas non-résolus
# ------------------------------------------------------------
print("\n🎯 ÉTAPE 8 : Analyse des cas non-résolus...")

# Identifier tous les cas non-résolus dans les données originales
unsolved_mask = df_clean['Solved'] == 0
unsolved_features = X_encoded[unsolved_mask]

print(f"Nombre de cas non-résolus à analyser: {unsolved_features.shape[0]}")

if unsolved_features.shape[0] > 0:
    # Prédire les probabilités de résolution
    unsolved_probs = model.predict_proba(unsolved_features)[:, 1]
    
    print(f"\n📊 STATISTIQUES DES PROBABILITÉS:")
    print(f"   Moyenne: {unsolved_probs.mean():.3f}")
    print(f"   Médiane: {np.median(unsolved_probs):.3f}")
    print(f"   Minimum: {unsolved_probs.min():.3f}")
    print(f"   Maximum: {unsolved_probs.max():.3f}")
    
    # Catégoriser les cas
    print(f"\n🔎 CATÉGORISATION DES CAS NON-RÉSOLUS:")
    thresholds = [
        (0.8, 1.0, "Très solubles"),
        (0.6, 0.8, "Potentiellement solubles"),
        (0.4, 0.6, "Difficiles"),
        (0.2, 0.4, "Très difficiles"),
        (0.0, 0.2, "Extrêmement difficiles")
    ]
    
    for low, high, label in thresholds:
        if high == 1.0:
            count = (unsolved_probs > low).sum()
        else:
            count = ((unsolved_probs >= low) & (unsolved_probs < high)).sum()
        percentage = count / len(unsolved_probs) * 100
        print(f"   {label}: {count} cas ({percentage:.1f}%)")
    
    # Identifier les cas à haut potentiel
    high_potential_threshold = 0.7
    high_potential_count = (unsolved_probs > high_potential_threshold).sum()
    print(f"\n🎯 CAS À HAUT POTENTIEL (prob > {high_potential_threshold}): {high_potential_count}")
    
    # Créer un DataFrame avec les résultats
    results_df = df_clean[unsolved_mask].copy()
    results_df['Predicted_Solvability'] = unsolved_probs
    results_df = results_df.sort_values('Predicted_Solvability', ascending=False)
    
    # Sauvegarder les résultats
    results_df.to_csv('cas_non_resolus_classifies.csv', index=False)
    print(f"\n💾 Résultats sauvegardés dans 'cas_non_resolus_classifies.csv'")
    
    # Afficher les 5 cas les plus prometteurs
    print(f"\n🏆 TOP 5 CAS LES PLUS PROMETTEURS:")
    for i in range(min(5, len(results_df))):
        case = results_df.iloc[i]
        solvability = case['Predicted_Solvability']
        print(f"\n{i+1}. Probabilité: {solvability:.1%}")
        print(f"   Arme: {case['Weapon']}")
        print(f"   Relation: {case['Relationship']}")
        print(f"   Victime: {case['Victim Sex']}, {case['Victim Age']} ans")
        print(f"   Suspect: {case['Perpetrator Sex']}, {case['Perpetrator Age']} ans")

# ------------------------------------------------------------
# FIN
# ------------------------------------------------------------
print("\n" + "="*60)
print("✨ ANALYSE TERMINÉE AVEC SUCCÈS !")
print("="*60)

print(f"""
📊 RÉCAPITULATIF FINAL:
• Total de cas analysés: {len(df)}
• Modèle: Random Forest (100 arbres)
• Cas non-résolus analysés: {unsolved_features.shape[0] if 'unsolved_features' in locals() else 0}
• Fichier de résultats: 'cas_non_resolus_classifies.csv'

💡 INSIGHTS CLÉS:
1. Les features de qualité de données sont cruciales
2. L'identification du suspect est le facteur #1
3. Certains cas non-résolus ont un haut potentiel de résolution

🚀 PROCHAINES ÉTAPES:
1. Examiner le fichier CSV généré
2. Prioriser les cas avec haute probabilité
3. Affiner le modèle avec les résultats
""")
print("\n" + "="*60)
print("📋 ÉTAPE 1 : EXAMEN DU FICHIER CSV GÉNÉRÉ")
print("="*60)

# Charger le fichier généré
try:
    results = pd.read_csv('cas_non_resolus_classifies.csv')
    print(f"✅ Fichier chargé : {len(results)} cas non-résolus")
    
    # Afficher les colonnes
    print("\n📊 COLONNES DISPONIBLES :")
    print(results.columns.tolist())
    
    # Statistiques descriptives
    print("\n📈 STATISTIQUES DES PROBABILITÉS :")
    print(results['Predicted_Solvability'].describe())
    
    # Distribution des probabilités
    plt.figure(figsize=(10, 6))
    plt.hist(results['Predicted_Solvability'], bins=50, alpha=0.7, 
             color='blue', edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Seuil 50%')
    plt.axvline(x=0.7, color='orange', linestyle='--', label='Seuil 70%')
    plt.xlabel('Probabilité de Résolution')
    plt.ylabel('Nombre de Cas')
    plt.title('Distribution des Probabilités de Résolution\nCas Non-Résolus')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Top 10 des cas les plus prometteurs
    print("\n🏆 TOP 10 DES CAS LES PLUS PROMETTEURS :")
    top_cases = results.nlargest(10, 'Predicted_Solvability')
    for i, (_, case) in enumerate(top_cases.iterrows(), 1):
        print(f"\n{i}. Probabilité: {case['Predicted_Solvability']:.1%}")
        print(f"   Record ID: {case.get('Record ID', 'N/A')}")
        print(f"   Année: {case.get('Year', 'N/A')}, État: {case.get('State', 'N/A')}")
        print(f"   Arme: {case.get('Weapon', 'N/A')}")
        print(f"   Relation: {case.get('Relationship', 'N/A')}")
        print(f"   Victime: {case.get('Victim Sex', 'N/A')} {case.get('Victim Age', 'N/A')} ans")
    
except FileNotFoundError:
    print("❌ Fichier 'cas_non_resolus_classifies.csv' non trouvé!")
    print("   Assurez-vous que le script précédent a bien généré le fichier.")
    print("\n" + "="*60)
print("🎯 ÉTAPE 2 : PRIORISATION DES CAS")
print("="*60)

# Définir des seuils de priorité
priority_thresholds = {
    'PRIORITÉ MAXIMALE (prob > 0.8)': 0.8,
    'HAUTE PRIORITÉ (prob 0.7-0.8)': 0.7,
    'PRIORITÉ MOYENNE (prob 0.5-0.7)': 0.5,
    'BASSE PRIORITÉ (prob < 0.5)': 0.0
}

print("\n📊 CATÉGORIES DE PRIORITÉ :")
print("-" * 40)

for priority_name, threshold in priority_thresholds.items():
    if threshold == 0.8:
        mask = results['Predicted_Solvability'] > threshold
    elif threshold == 0.0:
        mask = results['Predicted_Solvability'] < 0.5
    elif threshold == 0.7:
        mask = (results['Predicted_Solvability'] > threshold) & (results['Predicted_Solvability'] <= 0.8)
    else:  # 0.5
        mask = (results['Predicted_Solvability'] > threshold) & (results['Predicted_Solvability'] <= 0.7)
    
    count = mask.sum()
    percentage = count / len(results) * 100
    
    print(f"{priority_name}:")
    print(f"  Nombre de cas: {count} ({percentage:.1f}%)")
    
    if count > 0:
        # Caractéristiques de cette catégorie
        category_cases = results[mask]
        print(f"  Âge moyen victime: {category_cases['Victim Age'].mean():.1f} ans")
        print(f"  Arme la plus fréquente: {category_cases['Weapon'].mode().iloc[0] if not category_cases['Weapon'].mode().empty else 'N/A'}")
        print(f"  Relation la plus fréquente: {category_cases['Relationship'].mode().iloc[0] if not category_cases['Relationship'].mode().empty else 'N/A'}")
    print()

# Créer un fichier pour les enquêteurs
print("\n📁 CRÉATION DE FICHIERS PAR PRIORITÉ...")

# Fichier pour priorité maximale
max_priority = results[results['Predicted_Solvability'] > 0.8]
if len(max_priority) > 0:
    max_priority.to_csv('cas_priorite_maximale.csv', index=False)
    print(f"✅ 'cas_priorite_maximale.csv': {len(max_priority)} cas (prob > 80%)")

# Fichier pour haute priorité
high_priority = results[(results['Predicted_Solvability'] > 0.7) & (results['Predicted_Solvability'] <= 0.8)]
if len(high_priority) > 0:
    high_priority.to_csv('cas_haute_priorite.csv', index=False)
    print(f"✅ 'cas_haute_priorite.csv': {len(high_priority)} cas (prob 70-80%)")

# Rapport synthétique pour les enquêteurs
print("\n📄 CRÉATION D'UN RAPPORT SYNTHÉTIQUE...")

report_content = f"""
==================================================
RAPPORT DE PRIORISATION DES CAS NON-RÉSOLUS
Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
Total cas analysés: {len(results)}
==================================================

📊 RÉPARTITION PAR PRIORITÉ:
- Priorité Maximale (>80%): {len(max_priority)} cas
- Haute Priorité (70-80%): {len(high_priority)} cas
- Priorité Moyenne (50-70%): {len(results[(results['Predicted_Solvability'] > 0.5) & (results['Predicted_Solvability'] <= 0.7)])} cas
- Basse Priorité (<50%): {len(results[results['Predicted_Solvability'] <= 0.5])} cas

🔍 CARACTÉRISTIQUES DES CAS HAUTE PRIORITÉ:
1. Armes les plus fréquentes:
{results[results['Predicted_Solvability'] > 0.7]['Weapon'].value_counts().head(5).to_string()}

2. Relations les plus fréquentes:
{results[results['Predicted_Solvability'] > 0.7]['Relationship'].value_counts().head(5).to_string()}

3. Démographie victime:
- Âge moyen: {results[results['Predicted_Solvability'] > 0.7]['Victim Age'].mean():.1f} ans
- Sexe: {results[results['Predicted_Solvability'] > 0.7]['Victim Sex'].value_counts().to_string()}

💡 RECOMMANDATIONS:
1. Commencer par les {min(10, len(max_priority))} premiers cas de 'cas_priorite_maximale.csv'
2. Concentrer les ressources sur les affaires avec arme connue
3. Prioriser les cas avec relation victime-suspect identifiée
"""

# Sauvegarder le rapport
with open('rapport_priorisation.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("✅ 'rapport_priorisation.txt' créé")
print("\n📋 CONTENU DU RAPPORT :")
print(report_content)
print("\n" + "="*60)
print("🔄 ÉTAPE 3 : AFFINEMENT DU MODÈLE")
print("="*60)

# Stratégie 1 : Analyser les erreurs du modèle
print("\n1️⃣  ANALYSE DES ERREURS DU MODÈLE :")

# Identifier les faux négatifs (prédits non-résolus mais résolus)
print("\n🔎 FAUX NÉGATIFS (prédits non-résolus mais résolus) :")
# Note: Vous auriez besoin des prédictions sur l'ensemble complet pour cela
# Ceci est une suggestion pour l'avenir

# Stratégie 2 : Feature engineering additionnel
print("\n2️⃣  FEATURE ENGINEERING ADDITIONNEL :")

# Créer de nouvelles features basées sur les insights
df_refined = df_clean.copy()

# 1. Créer une feature d'interaction arme-relation
print("Création de nouvelles features...")

# Combinaison arme-relation (simplifiée)
def create_weapon_relation_feature(row):
    weapon = str(row['Weapon'])[:3] if pd.notna(row['Weapon']) else 'UNK'
    relation = str(row['Relationship'])[:3] if pd.notna(row['Relationship']) else 'UNK'
    return f"{weapon}_{relation}"

df_refined['Weapon_Relation_Combo'] = df_refined.apply(create_weapon_relation_feature, axis=1)

# 2. Créer des catégories d'âge
def age_category(age):
    if age < 18:
        return 'Juvenile'
    elif age < 30:
        return 'Young Adult'
    elif age < 50:
        return 'Adult'
    else:
        return 'Senior'

df_refined['Victim_Age_Category'] = df_refined['Victim Age'].apply(age_category)
df_refined['Perpetrator_Age_Category'] = df_refined['Perpetrator Age'].apply(age_category)

print(f"✅ Nouvelles features créées:")
print(f"   - Weapon_Relation_Combo: {df_refined['Weapon_Relation_Combo'].nunique()} catégories")
print(f"   - Victim_Age_Category: {df_refined['Victim_Age_Category'].unique()}")

# Stratégie 3 : Entraîner un modèle amélioré
print("\n3️⃣  ENTRAÎNEMENT D'UN MODÈLE AMÉLIORÉ :")

# Nouvelles features
new_features = available_features + ['Weapon_Relation_Combo', 'Victim_Age_Category', 'Perpetrator_Age_Category']
new_features = [f for f in new_features if f in df_refined.columns]

print(f"Features pour modèle amélioré: {len(new_features)}")
print(f"Liste: {new_features}")

# Préparation des données
X_new = df_refined[new_features]
X_new_encoded = pd.get_dummies(X_new, drop_first=True)

# Division
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new_encoded, y, test_size=0.3, random_state=42, stratify=y
)

# Nouveau modèle avec hyperparamètres optimisés
print("\nEntraînement du modèle amélioré...")
model_improved = RandomForestClassifier(
    n_estimators=200,      # Plus d'arbres
    max_depth=20,          # Plus profond
    min_samples_split=15,  # Paramètres ajustés
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model_improved.fit(X_train_new, y_train_new)

# Évaluation
y_pred_new = model_improved.predict(X_test_new)
y_pred_proba_new = model_improved.predict_proba(X_test_new)[:, 1]

from sklearn.metrics import roc_auc_score

roc_auc_new = roc_auc_score(y_test_new, y_pred_proba_new)
roc_auc_old = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 COMPARAISON DES PERFORMANCES :")
print(f"   Ancien modèle AUC-ROC: {roc_auc_old:.4f}")
print(f"   Nouveau modèle AUC-ROC: {roc_auc_new:.4f}")
print(f"   Amélioration: {(roc_auc_new - roc_auc_old):.4f}")

if roc_auc_new > roc_auc_old:
    print("   ✅ MODÈLE AMÉLIORÉ !")
else:
    print("   ⚠️  Modèle similaire ou moins bon")

# Stratégie 4 : Validation croisée
print("\n4️⃣  VALIDATION CROISÉE POUR ROBUSTESSE :")

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model_improved, 
    X_new_encoded, 
    y, 
    cv=5,  # 5 folds
    scoring='roc_auc',
    n_jobs=-1
)

print(f"Scores de validation croisée (AUC-ROC):")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"  Moyenne: {cv_scores.mean():.4f}")
print(f"  Écart-type: {cv_scores.std():.4f}")

# Stratégie 5 : Sauvegarde du modèle amélioré
print("\n5️⃣  SAUVEGARDE DU MODÈLE AMÉLIORÉ :")

import joblib

joblib.dump(model_improved, 'modele_ameliore.pkl')
joblib.dump(X_new_encoded.columns, 'features_ameliorees.pkl')

print("✅ Modèle sauvegardé dans 'modele_ameliore.pkl'")
print("✅ Features sauvegardées dans 'features_ameliorees.pkl'")

# Fonction pour prédire de nouveaux cas
def predict_new_case(case_data):
    """Prédire la solvabilité d'un nouveau cas"""
    model_loaded = joblib.load('modele_ameliore.pkl')
    features_loaded = joblib.load('features_ameliorees.pkl')
    
    # Préparer les données du nouveau cas
    # (À adapter selon votre structure)
    return "Fonction prête à être implémentée"

print("\n🔮 FONCTION DE PRÉDICTION PRÊTE POUR NOUVEAUX CAS")
print("\n" + "="*60)
print("📊 ÉTAPE 4 : CRÉATION DE VISUALISATIONS AVANCÉES")
print("="*60)

# 1. Heatmap des corrélations entre features importantes
print("\n1️⃣  HEATMAP DES CORRÉLATIONS :")

# Sélectionner les 20 features les plus importantes
top_20_features = feature_importance.head(20)['feature'].tolist()
if len(top_20_features) > 0:
    # Calculer la matrice de corrélation
    corr_matrix = X_encoded[top_20_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Corrélations entre les 20 Features les plus Importantes')
    plt.tight_layout()
    plt.show()

# 2. Analyse temporelle des probabilités
print("\n2️⃣  ANALYSE TEMPORELLE :")

if 'Year' in results.columns:
    plt.figure(figsize=(12, 6))
    
    # Moyenne des probabilités par année
    yearly_avg = results.groupby('Year')['Predicted_Solvability'].mean()
    
    plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2)
    plt.xlabel('Année')
    plt.ylabel('Probabilité moyenne de résolution')
    plt.title('Évolution des Probabilités de Résolution par Année\nCas Non-Résolus')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 3. Radar chart des caractéristiques des cas haute priorité
print("\n3️⃣  PROFIL DES CAS HAUTE PRIORITÉ :")

if len(max_priority) > 0:
    features_to_compare = ['Victim Age', 'Perpetrator Age', 'Victim Count', 'Perpetrator Count']
    features_to_compare = [f for f in features_to_compare if f in max_priority.columns]
    
    if features_to_compare:
        avg_values = max_priority[features_to_compare].mean()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        angles = np.linspace(0, 2*np.pi, len(features_to_compare), endpoint=False).tolist()
        values = avg_values.values.tolist()
        values += values[:1]  # Fermer le radar
        angles += angles[:1]
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features_to_compare)
        ax.set_title('Profil Moyen des Cas Haute Priorité')
        plt.tight_layout()
        plt.show()
        print("\n" + "="*60)
print("🔍 DIAGNOSTIC OVERFITTING - VÉRIFICATION RAPIDE")
print("="*60)

# 1. Comparaison Train/Test
from sklearn.model_selection import train_test_split

# Si vous n'avez pas déjà split, faites-le :
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

# Ré-entraînez sur train seulement
model.fit(X_train, y_train)

# Comparez les performances
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\n📊 COMPARAISON TRAIN/TEST:")
print(f"   Accuracy sur données d'entraînement: {train_acc:.4f} ({train_acc:.2%})")
print(f"   Accuracy sur données de test: {test_acc:.4f} ({test_acc:.2%})")
print(f"   Écart (overfitting): {train_acc - test_acc:.4f}")

# 2. Validation croisée simple
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_encoded, y, cv=5, scoring='accuracy')
print(f"\n🎯 VALIDATION CROISÉE (5 folds):")
print(f"   Scores: {cv_scores}")
print(f"   Moyenne: {cv_scores.mean():.4f} ({cv_scores.mean():.2%})")
print(f"   Écart-type: {cv_scores.std():.4f}")

# 3. Diagnostic
print(f"\n📈 DIAGNOSTIC FINAL:")
if train_acc - test_acc > 0.05:  # Plus de 5% d'écart
    print(f"   🔴 OVERFITTING SÉVÈRE DÉTECTÉ!")
    print(f"   → Votre modèle mémorise les données d'entraînement")
    print(f"   → Accuracy réelle estimée: {cv_scores.mean():.2%}")
elif train_acc > 0.99:
    print(f"   🟡 OVERFITTING MODÉRÉ - 99%+ est suspect")
    print(f"   → Performance probable: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
else:
    print(f"   🟢 MODÈLE CORRECT - Performance crédible")

