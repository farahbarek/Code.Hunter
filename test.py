"""
Script de test complet du modèle Random Forest pour prédiction de résolution d'homicides.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("PRÉDICTEUR DE SOLVABILITÉ D'HOMICIDES - ANALYSE COMPLÈTE")
print("=" * 70)

# ============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================
print("\n📂 Chargement des données...")

try:
    df = pd.read_csv(r"C:\homicide_data_mining\database.csv")
    print(f"✅ {len(df):,} cas chargés")
except FileNotFoundError:
    print("❌ Erreur : database.csv non trouvé")
    exit()

# Conversion variable cible
df['Solved'] = df['Crime Solved'].map({'Yes': 1, 'No': 0})

print(f"\n📊 Distribution :")
print(f"   Total : {len(df):,} cas")
print(f"   Résolus : {df['Solved'].sum():,} ({df['Solved'].mean():.1%})")
print(f"   Non-résolus : {(df['Solved'] == 0).sum():,}")

# ============================================================================
# ANALYSE EXPLORATOIRE
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSE DES FACTEURS")
print("=" * 70)

# Armes
if 'Weapon' in df.columns:
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

    df['Weapon_Group'] = df['Weapon'].apply(weapon_group)
    
    print("\n🔫 Solve rates par arme :")
    weapon_solve = df.groupby('Weapon_Group')['Solved'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    for weapon, row in weapon_solve.iterrows():
        print(f"   {weapon} : {row['mean']:.1%} ({row['count']:,} cas)")

# États
if 'State' in df.columns:
    print("\n🗺️  Top 5 États :")
    state_solve = df.groupby('State')['Solved'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    for i, (state, row) in enumerate(state_solve.head(5).iterrows(), 1):
        print(f"   {i}. {state} : {row['mean']:.1%}")

# ============================================================================
# PRÉPARATION POUR LE MODÈLE
# ============================================================================
print("\n" + "=" * 70)
print("PRÉPARATION DES DONNÉES")
print("=" * 70)

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

if 'Victim Age' in df.columns:
    df['Victim_Age_Category'] = df['Victim Age'].apply(categorize_age)

if 'State' in df.columns:
    state_solve_rates = df.groupby('State')['Solved'].mean().to_dict()
    df['State_Historical_Solve_Rate'] = df['State'].map(state_solve_rates)

# Sélection des features
key_features = [
    'Victim Sex', 'Victim Age', 'Victim Race',
    'Victim_Age_Category', 'Weapon_Group',
    'Victim Count', 'Year', 'State',
    'State_Historical_Solve_Rate'
]

available_features = [f for f in key_features if f in df.columns]
print(f"\n✅ {len(available_features)} features disponibles")

X = df[available_features].copy()
y = df['Solved'].copy()

# Encodage intelligent
for col in X.select_dtypes(include=['object']).columns:
    top_cats = X[col].value_counts().head(15).index
    X[col] = X[col].where(X[col].isin(top_cats), 'Other')

X_encoded = pd.get_dummies(X, drop_first=True)
print(f"✅ {X_encoded.shape[1]} features après encodage")

# ============================================================================
# DIVISION ET ENTRAÎNEMENT
# ============================================================================
print("\n" + "=" * 70)
print("ENTRAÎNEMENT DU MODÈLE")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n📊 Division :")
print(f"   Train : {X_train.shape[0]:,} ({y_train.mean():.1%})")
print(f"   Test : {X_test.shape[0]:,} ({y_test.mean():.1%})")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Modèle entraîné")

# ============================================================================
# ÉVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("ÉVALUATION")
print("=" * 70)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n📊 AUC-ROC : {auc:.4f}")

print("\n📋 Classification :")
print(classification_report(y_test, y_pred, target_names=['Non-Résolu', 'Résolu']))

cv_scores = cross_val_score(model, X_encoded, y, cv=5, scoring='roc_auc')
print(f"\n✔️  Validation croisée : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================================
# IMPORTANCE DES FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("TOP 10 FEATURES")
print("=" * 70)

feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i in range(min(10, len(feature_importance))):
    feat = feature_importance.iloc[i]
    print(f"{i+1:2}. {feat['feature'][:45]:45} : {feat['importance']:.4f}")

# ============================================================================
# PRÉDICTION CAS NON-RÉSOLUS
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

    # Sauvegarder résultats
    results_df = df[unsolved_mask].copy()
    results_df['Predicted_Solvability'] = unsolved_probs
    results_df = results_df.sort_values('Predicted_Solvability', ascending=False)
    results_df.to_csv('cas_non_resolus_classifies.csv', index=False)
    print(f"\n💾 Résultats sauvegardés")

    # Top 5 cas
    print(f"\n🏆 Top 5 :")
    for i in range(min(5, len(results_df))):
        case = results_df.iloc[i]
        print(f"\n{i+1}. Proba : {case['Predicted_Solvability']:.1%}")
        if 'Weapon' in case:
            print(f"   Arme : {case['Weapon']}")
        if 'Victim Sex' in case and 'Victim Age' in case:
            print(f"   Victime : {case['Victim Sex']}, {case['Victim Age']} ans")

# Sauvegarde du modèle
import joblib
import json

joblib.dump(model, 'homicide_model_optimized.pkl')
with open('model_features_optimized.json', 'w') as f:
    json.dump(list(X_encoded.columns), f)

print("\n\n" + "=" * 70)
print("✅ ANALYSE TERMINÉE")
print("=" * 70)


print("\n LOADING DATA (with sample for speed)...")
df = pd.read_csv(r"C:\homicide_data_mining\database.csv")

# TAKE SAMPLE FOR FASTER DEVELOPMENT - REMOVE THIS FOR FINAL RUN
sample_fraction = 0.3  # Use 30% for speed, change to 1.0 for final
df = df.sample(frac=sample_fraction, random_state=42)
print(f"✅ Using {len(df):,} samples ({sample_fraction*100:.0f}% of original)")
# PUT THE DATA ANALYSIS SECTION RIGHT HERE (AFTER LOADING, BEFORE PREPROCESSING)
print("\n" + "="*60)
print("🔍 DATA ANALYSIS - REALISTIC APPROACH")
print("="*60)
print("\n TARGET VARIABLE ANALYSIS:")

print("Crime Solved value counts:")
print(df['Crime Solved'].value_counts())

# Convert to binary target
df['Solved'] = df['Crime Solved'].map({'Yes': 1, 'No': 0})

print(f"\nTotal cases: {len(df):,}")
print(f"Solved cases: {df['Solved'].sum():,} ({df['Solved'].mean():.2%})")
print(f"Unsolved cases: {len(df) - df['Solved'].sum():,} ({1-df['Solved'].mean():.2%})")


print("\n👤 VICTIM CHARACTERISTICS ANALYSIS:")

victim_features = ['Victim Sex', 'Victim Age', 'Victim Race', 'Victim Ethnicity']

for feature in victim_features:
    if feature in df.columns:
        print(f"\n--- {feature} ---")
        
        # Calculate solve rates by category
        solve_rates = df.groupby(feature)['Solved'].mean().sort_values(ascending=False)
        
        for category, rate in solve_rates.head(5).items():
            count = df[df[feature] == category].shape[0]
            print(f"  {category}: {rate:.1%} solved ({count:,} cases)")
        
        if len(solve_rates) > 5:
            print(f"  ... and {len(solve_rates) - 5} other categories")


print("\n WEAPON ANALYSIS:")

if 'Weapon' in df.columns:
    # Group similar weapons
    def weapon_group(weapon):
        if pd.isna(weapon):
            return 'Unknown'
        weapon_lower = str(weapon).lower()
        if 'firearm' in weapon_lower or 'gun' in weapon_lower:
            return 'Firearm'
        elif 'knife' in weapon_lower or 'cut' in weapon_lower:
            return 'Knife'
        elif 'blunt' in weapon_lower:
            return 'Blunt Object'
        elif 'strangulation' in weapon_lower:
            return 'Strangulation'
        else:
            return 'Other'
    
    df['Weapon_Group'] = df['Weapon'].apply(weapon_group)
    
    print("\nSolve rates by weapon type:")
    weapon_solve = df.groupby('Weapon_Group')['Solved'].agg(['mean', 'count'])
    weapon_solve = weapon_solve.sort_values('mean', ascending=False)
    
    for weapon_type, row in weapon_solve.iterrows():
        print(f"  {weapon_type}: {row['mean']:.1%} solved ({row['count']:,} cases)")


print("\n TEMPORAL ANALYSIS:")

if 'Year' in df.columns:
    print("\nSolve rates by year:")
    yearly_solve = df.groupby('Year')['Solved'].mean()
    
    # Get top and bottom years
    print("  Best years for solving:")
    for year, rate in yearly_solve.nlargest(3).items():
        print(f"    {year}: {rate:.1%}")
    
    print("  Worst years for solving:")
    for year, rate in yearly_solve.nsmallest(3).items():
        print(f"    {year}: {rate:.1%}")
    
    print(f"  Overall trend: {yearly_solve.min():.1%} to {yearly_solve.max():.1%}")

print("\n GEOGRAPHICAL ANALYSIS:")

if 'State' in df.columns:
    print("\nTop 5 states by solve rate:")
    state_solve = df.groupby('State')['Solved'].mean().sort_values(ascending=False)
    
    for state, rate in state_solve.head(5).items():
        count = df[df['State'] == state].shape[0]
        print(f"  {state}: {rate:.1%} solved ({count:,} cases)")
    
    print("\nBottom 5 states by solve rate:")
    for state, rate in state_solve.tail(5).items():
        count = df[df['State'] == state].shape[0]
        print(f"  {state}: {rate:.1%} solved ({count:,} cases)")


print("\n DATA QUALITY CHECK (Safe Features):")

safe_features_to_check = ['Victim Age', 'Victim Sex', 'Weapon', 'Year', 'State']

missing_data = df[safe_features_to_check].isnull().mean() * 100
missing_data = missing_data[missing_data > 0]

if len(missing_data) > 0:
    print("Features with missing data:")
    for feature, missing_pct in missing_data.items():
        print(f"  {feature}: {missing_pct:.1f}% missing")
else:
    print("✅ No missing data in key safe features")

print("\n" + "="*60)
print(" INVESTIGATION INSIGHTS FROM DATA")
print("="*60)

print(f"""
 KEY FINDINGS FROM INITIAL DATA:

1. SOLVE RATE: {df['Solved'].mean():.1%} of cases are solved

2. VICTIM FACTORS:
   • Age groups with different solve rates
   • Demographic patterns in solved cases

3. WEAPON IMPACT:
   • Firearms vs. knives vs. other weapons
   • Different solve rates by weapon type

4. GEOGRAPHICAL PATTERNS:
   • State-by-state variation in solve rates
   • Jurisdictional differences

5. TEMPORAL TRENDS:
   • Year-to-year changes in solve rates
   • Seasonal patterns (if month data available)

""")



print("\n" + "="*60)
print(" OPTIMIZED PREPROCESSING...")
print("="*60)

# Create a clean copy
df_clean = df.copy()

# A. Handle Weapon - replace Unknown with mode
if 'Weapon' in df_clean.columns:
    weapon_mode = df_clean['Weapon'].mode()[0] if not df_clean['Weapon'].mode().empty else 'Unknown'
    df_clean['Weapon'] = df_clean['Weapon'].replace('Unknown', weapon_mode)

# B. Create temporal features
if 'Year' in df_clean.columns:
    df_clean['Year_Normalized'] = (df_clean['Year'] - df_clean['Year'].min()) / (df_clean['Year'].max() - df_clean['Year'].min())

if 'Month' in df_clean.columns:
    # Convert month to numerical
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df_clean['Month_Numeric'] = df_clean['Month'].map(month_map)
    df_clean['Is_Weekend_Month'] = df_clean['Month_Numeric'].apply(
        lambda x: 1 if x in [1, 7, 12] else 0
    )

# C. Create age categories for victim
def categorize_age(age):
    if pd.isna(age):
        return 'Unknown'
    try:
        age = float(age)
        if age < 18:
            return 'Child'
        elif age < 30:
            return 'Young Adult'
        elif age < 50:
            return 'Adult'
        elif age < 65:
            return 'Middle Aged'
        else:
            return 'Senior'
    except:
        return 'Unknown'

df_clean['Victim_Age_Category'] = df_clean['Victim Age'].apply(categorize_age)

# D. Feature engineering - REDUCED FOR SPEED
def weapon_category(weapon):
    if pd.isna(weapon):
        return 'Unknown'
    weapon_lower = str(weapon).lower()
    if 'firearm' in weapon_lower or 'gun' in weapon_lower:
        return 'Firearm'
    elif 'knife' in weapon_lower:
        return 'Knife'
    else:
        return 'Other'

df_clean['Weapon_Category'] = df_clean['Weapon'].apply(weapon_category)

# E. State-level solve rates
state_solve_rates = df.groupby('State')['Solved'].mean().to_dict()
df_clean['State_Historical_Solve_Rate'] = df_clean['State'].map(state_solve_rates)

# -----------------------------------------------------------------
# 8. FINAL FEATURE SELECTION - REDUCED FOR SPEED
# -----------------------------------------------------------------
print("\n SELECTING KEY FEATURES FOR SPEED...")

# Use only most important features for faster training
key_features = [
    # Victim demographics (most important)
    'Victim Sex', 'Victim Age', 'Victim Race',
    'Victim_Age_Category',
    
    # Crime details
    'Weapon_Category', 'Victim Count',
    
    # Time/Location (simplified)
    'Year', 'State', 'State_Historical_Solve_Rate',
]

available_features = [f for f in key_features if f in df_clean.columns]
X = df_clean[available_features]
y = df_clean['Solved']

print(f"   Using {len(available_features)} key features (instead of 16)")
print(f"   Features: {available_features}")

# -----------------------------------------------------------------
# 9. SMART ENCODING - REDUCE DIMENSIONALITY
# -----------------------------------------------------------------
print("\n SMART ENCODING (reducing dimensions)...")

# Limit categorical values before encoding
for col in X.select_dtypes(include=['object']).columns:
    # Keep only top 15 categories, group rest as 'Other'
    top_cats = X[col].value_counts().head(15).index
    X[col] = X[col].where(X[col].isin(top_cats), 'Other')

X_encoded = pd.get_dummies(X, drop_first=True)
print(f"   After smart encoding: {X_encoded.shape[1]} features (was ~1800+)")

# -----------------------------------------------------------------
# 10. FAST MODEL TRAINING
# -----------------------------------------------------------------
print("\n" + "="*60)
print(" FAST RANDOM FOREST TRAINING")
print("="*60)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

print(f" Data Split:")
print(f"   Train: {X_train.shape[0]:,} samples ({y_train.mean():.2%} solved)")
print(f"   Test:  {X_test.shape[0]:,} samples ({y_test.mean():.2%} solved)")

# OPTIMIZED model for speed
model = RandomForestClassifier(
    n_estimators=50,           # Reduced from 100
    max_depth=8,               # Reduced from 10
    min_samples_split=100,     # Increased for speed
    min_samples_leaf=50,       # Increased for speed
    max_features=0.3,          # Reduced from 0.5
    bootstrap=True,
    oob_score=True,            # Get OOB score for validation
    random_state=42,
    n_jobs=-1,                 # Use all CPU cores
    verbose=1                  # Show progress
)

print("\n TRAINING MODEL (This will be faster)...")
model.fit(X_train, y_train)
print(" Model training complete!")

# Get predictions
y_pred_train = model.predict_proba(X_train)[:, 1]
y_pred_test = model.predict_proba(X_test)[:, 1]

# Calculate AUC
train_auc = roc_auc_score(y_train, y_pred_train)
test_auc = roc_auc_score(y_test, y_pred_test)

print(f"\n AUC SCORES:")
print(f"   Train AUC: {train_auc:.4f}")
print(f"   Test AUC:  {test_auc:.4f}")
print(f"   OOB Score: {model.oob_score_:.4f}")
print(f"   Overfitting gap: {train_auc - test_auc:.4f}")

# -----------------------------------------------------------------
# 11. CLASSIFICATION METRICS
# -----------------------------------------------------------------
print("\n DETAILED CLASSIFICATION METRICS:")

# Convert probabilities to binary predictions
threshold = 0.5
y_pred_binary = (y_pred_test >= threshold).astype(int)

print(f"\n CLASSIFICATION REPORT (threshold = {threshold}):")
print(classification_report(y_test, y_pred_binary, 
                          target_names=['Unsolved', 'Solved'],
                          digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
tn, fp, fn, tp = cm.ravel()

print(f"\n CONFUSION MATRIX:")
print(f"               Predicted Unsolved   Predicted Solved")
print(f"Actual Unsolved       {tn:>10}           {fp:>10}")
print(f"Actual Solved         {fn:>10}           {tp:>10}")

# Calculate key metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n KEY METRICS:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
print(f"   Precision: {precision:.4f} ({precision:.2%})")
print(f"   Recall:    {recall:.4f} ({recall:.2%})")
print(f"   F1-Score:  {f1:.4f}")

# -----------------------------------------------------------------
# 12. FEATURE IMPORTANCE
# -----------------------------------------------------------------
print("\n🔍 TOP 10 MOST IMPORTANT FEATURES:")

feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i in range(min(10, len(feature_importance))):
    feat = feature_importance.iloc[i]
    feature_name = feat['feature'][:40]
    print(f"{i+1:2}. {feature_name:40} : {feat['importance']:.4f}")


print("\n" + "="*60)
print("💰 BUSINESS IMPACT ANALYSIS")
print("="*60)

print(f"\n IN THE TEST SET:")
print(f"   Total cases: {len(y_test):,}")
print(f"   Actually solved: {y_test.sum():,} ({y_test.mean():.1%})")
print(f"   Actually unsolved: {len(y_test) - y_test.sum():,} ({(1-y_test.mean()):.1%})")

# Model prioritization analysis
print(f"\n MODEL PRIORITIZATION ANALYSIS:")

# How many cases would we prioritize?
for threshold, label in [(0.8, "High Priority"), (0.7, "Medium Priority"), (0.6, "Low Priority")]:
    prioritized = (y_pred_test >= threshold).sum()
    prioritized_pct = prioritized / len(y_test) * 100
    
    # Of those prioritized, how many are actually solvable?
    true_positives = ((y_pred_test >= threshold) & (y_test == 1)).sum()
    if prioritized > 0:
        precision_at_threshold = true_positives / prioritized
    else:
        precision_at_threshold = 0
    
    print(f"\n   {label} (prob > {threshold}):")
    print(f"   • Would prioritize: {prioritized:,} cases ({prioritized_pct:.1f}% of total)")
    print(f"   • Actually solvable: {true_positives:,} ({precision_at_threshold:.1%} precision)")
    print(f"   • Would catch: {true_positives/y_test.sum()*100:.1f}% of all solvable cases")

# -----------------------------------------------------------------
# 14. ANALYZE UNSOLVED CASES
# -----------------------------------------------------------------
print("\n" + "="*60)
print(" ANALYZING UNSOLVED CASES FOR PRIORITIZATION")
print("="*60)

# Create a results dataframe for test cases
test_results_list = []

for i, idx in enumerate(X_test.index):
    # Get prediction for this case
    prob = y_pred_test[i]
    actual = y_test.iloc[i]
    
    # Get original features for this case
    case_data = {
        'Index': idx,
        'Actual_Solved': actual,
        'Predicted_Probability': prob,
        'Predicted_Class': 1 if prob >= 0.5 else 0
    }
    
    # Add safe features from original data
    safe_features = ['Victim Sex', 'Victim Age', 'Victim Race', 
                    'Weapon', 'State', 'Year', 'Victim Count']
    
    for feature in safe_features:
        if feature in df_clean.columns and idx in df_clean.index:
            case_data[feature] = df_clean.loc[idx, feature]
    
    test_results_list.append(case_data)

# Convert to DataFrame
test_results = pd.DataFrame(test_results_list)

print(f" Test set analysis:")
print(f"   Total test cases analyzed: {len(test_results):,}")
print(f"   Actually solved: {test_results['Actual_Solved'].sum():,} ({test_results['Actual_Solved'].mean():.1%})")
print(f"   Actually unsolved: {(test_results['Actual_Solved'] == 0).sum():,} ({(test_results['Actual_Solved'] == 0).mean():.1%})")

# Analyze unsolved cases
unsolved_cases = test_results[test_results['Actual_Solved'] == 0].copy()

if len(unsolved_cases) > 0:
    # Find high-potential unsolved cases
    high_potential = unsolved_cases[unsolved_cases['Predicted_Probability'] > 0.7]
    
    print(f"\n High-potential unsolved cases (>70% predicted): {len(high_potential):,}")
    
    if len(high_potential) > 0:
        print(f"\n TOP 5 HIGH-POTENTIAL UNSOLVED CASES:")
        top_unsolved = high_potential.nlargest(5, 'Predicted_Probability')
        
        for i, (_, case) in enumerate(top_unsolved.iterrows(), 1):
            print(f"\n{i}. Predicted Solvability: {case['Predicted_Probability']:.1%}")
            if 'Victim Sex' in case and pd.notna(case['Victim Sex']):
                print(f"   Victim: {case['Victim Sex']}, {case.get('Victim Age', 'Unknown')} years")
            if 'Weapon' in case and pd.notna(case['Weapon']):
                print(f"   Weapon: {case['Weapon']}")
            if 'State' in case and pd.notna(case['State']):
                print(f"   State: {case['State']}")
            if 'Year' in case and pd.notna(case['Year']):
                print(f"   Year: {case['Year']}")

print("\n" + "="*60)
print(" SAVING MODEL AND ARTIFACTS")
print("="*60)

import joblib
import json
from datetime import datetime

# Save model
model_filename = 'homicide_model_optimized.pkl'
joblib.dump(model, model_filename)

# Save feature names
feature_filename = 'model_features_optimized.json'
with open(feature_filename, 'w') as f:
    json.dump(list(X_encoded.columns), f)

print(f" Model saved: {model_filename}")
print(f" Features saved: {feature_filename}")

# -----------------------------------------------------------------
# 16. PREDICTION FUNCTION
# -----------------------------------------------------------------
print("\n" + "="*60)
print(" PREDICTION FUNCTION FOR NEW CASES")
print("="*60)

def predict_solvability(case_data, model_path='homicide_model_optimized.pkl', 
                       features_path='model_features_optimized.json'):
    """
    Predict solvability probability for a new homicide case
    """
    
    print("\n PROCESSING NEW CASE...")
    
    try:
        # Load model and features
        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            expected_features = json.load(f)
        
        print(f" Model and features loaded successfully")
        
        # Prepare case data
        case_df = pd.DataFrame([case_data])
        
        # Apply preprocessing
        case_df['Victim_Age_Category'] = case_df['Victim Age'].apply(categorize_age)
        
        # Ensure Weapon_Category exists
        if 'Weapon_Category' not in case_df.columns and 'Weapon' in case_df.columns:
            case_df['Weapon_Category'] = case_df['Weapon'].apply(weapon_category)
        
        # Fill missing State_Historical_Solve_Rate
        if 'State_Historical_Solve_Rate' not in case_df.columns and 'State' in case_df.columns:
            case_df['State_Historical_Solve_Rate'] = case_df['State'].map(state_solve_rates)
            # If state not in training data, use average
            if case_df['State_Historical_Solve_Rate'].isna().any():
                case_df['State_Historical_Solve_Rate'] = df['Solved'].mean()
        
        # Select only the features we used in training
        case_features = case_df[available_features].copy()
        
        # Apply same categorical grouping
        for col in case_features.select_dtypes(include=['object']).columns:
            if col in X.columns:
                top_cats = X[col].value_counts().head(15).index
                case_features[col] = case_features[col].apply(
                    lambda x: x if x in top_cats else 'Other'
                )
        
        # One-hot encode
        case_encoded = pd.get_dummies(case_features, drop_first=True)
        
        # Align columns with training data
        for col in expected_features:
            if col not in case_encoded.columns:
                case_encoded[col] = 0
        
        case_encoded = case_encoded[expected_features]
        
        # Make prediction
        probability = model.predict_proba(case_encoded)[0, 1]
        
        # Determine priority level
        if probability >= 0.8:
            priority = "HIGH PRIORITY"
            recommendation = "Assign experienced investigators immediately"
        elif probability >= 0.7:
            priority = "MEDIUM-HIGH PRIORITY"
            recommendation = "Assign dedicated investigator"
        elif probability >= 0.6:
            priority = "MEDIUM PRIORITY"
            recommendation = "Standard investigation"
        elif probability >= 0.5:
            priority = "LOW-MEDIUM PRIORITY"
            recommendation = "Basic investigation"
        else:
            priority = "LOW PRIORITY"
            recommendation = "Minimal resources, revisit if new evidence emerges"
        
        result = {
            'case_id': case_data.get('case_id', 'Unknown'),
            'solvability_probability': float(probability),
            'solvability_percentage': f"{probability:.1%}",
            'priority_level': priority,
            'predicted_outcome': 'Likely Solvable' if probability >= 0.5 else 'Likely Unsolvable',
            'investigation_recommendation': recommendation,
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        print(f" Prediction complete!")
        return result
        
    except Exception as e:
        print(f" Error in prediction: {e}")
        return {
            'error': str(e),
            'prediction_timestamp': datetime.now().isoformat()
        }

# -----------------------------------------------------------------
# 17. TEST PREDICTIONS
# -----------------------------------------------------------------
print("\n" + "="*60)
print(" TESTING PREDICTION FUNCTION")
print("="*60)

# Define sample cases
sample_case_1 = {
    'case_id': 'TEST-001',
    'Victim Sex': 'Male',
    'Victim Age': 35,
    'Victim Race': 'White',
    'Weapon': 'Handgun',
    'Weapon_Category': 'Firearm',
    'State': 'California',
    'Year': 2020,
    'Victim Count': 1
}

sample_case_2 = {
    'case_id': 'TEST-002',
    'Victim Sex': 'Female',
    'Victim Age': 25,
    'Victim Race': 'Black',
    'Weapon': 'Unknown',
    'Weapon_Category': 'Other',
    'State': 'Alabama',
    'Year': 2019,
    'Victim Count': 1
}

# Test predictions
print("\n SAMPLE CASE 1 (Firearm in California):")
result1 = predict_solvability(sample_case_1)
if 'error' not in result1:
    print(f"   Case ID: {result1['case_id']}")
    print(f"   Solvability: {result1['solvability_percentage']}")
    print(f"   Priority: {result1['priority_level']}")
    print(f"   Recommendation: {result1['investigation_recommendation']}")

print("\n SAMPLE CASE 2 (Unknown weapon in Alabama):")
result2 = predict_solvability(sample_case_2)
if 'error' not in result2:
    print(f"   Case ID: {result2['case_id']}")
    print(f"   Solvability: {result2['solvability_percentage']}")
    print(f"   Priority: {result2['priority_level']}")
    print(f"   Recommendation: {result2['investigation_recommendation']}")


print("\n" + "="*60)
print(" CREATING VISUALIZATIONS")
print("="*60)

try:
    # 1. Feature Importance Bar Chart
    print("\n1️  FEATURE IMPORTANCE VISUALIZATION:")
    plt.figure(figsize=(10, 6))
    top_n = 10
    top_features = feature_importance.head(top_n)
    
    plt.barh(range(top_n), top_features['importance'])
    plt.yticks(range(top_n), top_features['feature'], fontsize=9)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features Influencing Solvability')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("    Saved: feature_importance.png")
    
    # 2. ROC Curve
    print("\n2  ROC CURVE VISUALIZATION:")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'AUC = {test_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Homicide Solvability Predictor')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("    Saved: roc_curve.png")
    
    # 3. Probability Distribution
    print("\n3️ PROBABILITY DISTRIBUTION:")
    plt.figure(figsize=(10, 6))
    
    # Plot distribution for solved vs unsolved
    plt.hist(y_pred_test[y_test == 1], bins=50, alpha=0.7, 
             label='Actually Solved', color='green', density=True)
    plt.hist(y_pred_test[y_test == 0], bins=50, alpha=0.7, 
             label='Actually Unsolved', color='red', density=True)
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary')
    plt.xlabel('Predicted Solvability Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Probabilities\n(Solved vs Unsolved Cases)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('probability_distribution.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("    Saved: probability_distribution.png")
    
except Exception as e:
    print(f"   Visualization error: {e}")
    print("   Continuing without visualizations...")

# 19. FINAL SUMMARY

print("\n" + "="*60)
print("✨ FINAL SUMMARY")
print("="*60)

print(f"""
 KEY FINDINGS:

1. MODEL PERFORMANCE:
   • Test AUC: {test_auc:.3f} ({(test_auc-0.5)*100:.1f}% better than random)
   • OOB Score: {model.oob_score_:.3f}
   • Accuracy: {accuracy:.1%}

2. BUSINESS IMPACT:
   • Model can prioritize cases effectively
   • Prediction function ready for new cases
   • Provides actionable recommendations
 FILES CREATED:
• homicide_model_optimized.pkl
• model_features_optimized.json
• feature_importance.png
• roc_curve.png
• probability_distribution.png


""")

print("\n" + "="*60)
print(" HOMICIDE SOLVABILITY PREDICTOR - COMPLETE!")
print("="*60)