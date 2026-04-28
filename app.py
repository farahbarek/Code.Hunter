from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import json
import os
import traceback

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "homicide_data_mining")
MODEL_PATH = os.path.join(DATA_DIR, "homicide_model_optimized.pkl")
FEATURES_PATH = os.path.join(DATA_DIR, "model_features_optimized.json")

# Chargement du modèle au démarrage
print("=" * 60)
print("DÉMARRAGE DU SERVEUR IA")
print("=" * 60)

model = None
expected_features = None

try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        expected_features = json.load(f)
    print(f"✅ Modèle chargé : {len(expected_features)} features")
except Exception as e:
    print(f"❌ Erreur : {e}")
    model = None


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def categorize_age(age):
    """Catégorise l'âge en groupes."""
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
    except (ValueError, TypeError):
        return 'Unknown'


def weapon_category(weapon):
    """Catégorise les armes en groupes principaux."""
    try:
        w = str(weapon).lower()
        if any(k in w for k in ['firearm', 'gun', 'handgun', 'shotgun', 'rifle']):
            return 'Firearm'
        elif any(k in w for k in ['knife', 'cut', 'stab']):
            return 'Knife'
        elif 'blunt' in w:
            return 'Blunt Object'
        elif 'strangulation' in w:
            return 'Strangulation'
        else:
            return 'Other'
    except:
        return 'Other'


def is_vulnerable(age):
    """Détermine si une victime est vulnérable (enfant ou senior)."""
    try:
        a = float(age)
        return a < 18 or a > 65
    except (ValueError, TypeError):
        return False


# ============================================================================
# ENDPOINTS API
# ============================================================================

@app.route('/api/features', methods=['GET'])
def get_features():
    """Retourne la liste des features attendues par le modèle."""
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500

    return jsonify({
        "features_count": len(expected_features),
        "features": expected_features,
        "model_status": "ready"
    })


@app.route('/')
def index():
    """Serve la page HTML."""
    return send_from_directory('.', 'crime_predictor.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Analyse les données de cas et prédit la probabilité de résolution."""
    if model is None:
        return jsonify({"error": "Le modèle IA n'est pas chargé."}), 500

    try:
        data = request.json
        print(f"\n🔍 Analyse : {data.get('state')} / {data.get('weapon')}")

        # Validation et conversion des données
        try:
            v_age = float(data.get('age', 0)) if data.get('age') else 25.0
            v_count = int(data.get('victims', 1)) if data.get('victims') else 1
            v_year = int(data.get('year', 2023)) if data.get('year') else 2023
            v_rate = float(data.get('historicalRate', 0.6))
        except ValueError:
            return jsonify({"error": "Données numériques invalides"}), 400

        # Préparation du DataFrame
        case_df = pd.DataFrame([{
            'Victim Sex': data.get('sex', 'Unknown'),
            'Victim Age': v_age,
            'Victim Race': data.get('race', 'Unknown'),
            'Weapon': data.get('weapon', 'Other'),
            'State': data.get('state', 'Unknown'),
            'Year': v_year,
            'Victim Count': v_count,
            'State_Historical_Solve_Rate': v_rate
        }])

        # Feature Engineering
        case_df['Victim_Age_Category'] = case_df['Victim Age'].apply(categorize_age)
        case_df['Weapon_Category'] = case_df['Weapon'].apply(weapon_category)

        # Sélection des features
        key_features = [
            'Victim Sex', 'Victim Age', 'Victim Race',
            'Victim_Age_Category', 'Weapon_Category',
            'Victim Count', 'Year', 'State',
            'State_Historical_Solve_Rate'
        ]

        case_features = case_df[[f for f in key_features if f in case_df.columns]]
        case_encoded = pd.get_dummies(case_features)

        # Alignement avec le modèle
        for col in expected_features:
            if col not in case_encoded.columns:
                case_encoded[col] = 0

        final_df = case_encoded[expected_features]

        # Prédiction
        prob = float(model.predict_proba(final_df)[0, 1])
        verdict = "Resolved" if prob >= 0.55 else "Unresolved"
        v_status = is_vulnerable(v_age)
        v_label = "Vulnérable" if v_status else "Adulte Standard"

        print(f"   Probabilité : {prob:.2%}")
        print(f"   Verdict : {verdict}")

        return jsonify({
            "probability": prob,
            "probability_percent": f"{prob*100:.1f}%",
            "verdict": verdict,
            "vulnerability": v_status,
            "vulnerability_label": v_label,
            "key_factors": [
                {
                    "label": "Fiabilité Modèle",
                    "value": "Random Forest Optimisé (AUC ~0.82)",
                    "impact": "positive"
                },
                {
                    "label": "Facteur Arme",
                    "value": f"Impact {case_df['Weapon_Category'][0]}",
                    "impact": "neutral" if case_df['Weapon_Category'][0] == 'Other' else "positive"
                },
                {
                    "label": "Démographie",
                    "value": f"Victime {case_df['Victim_Age_Category'][0]} ({v_label})",
                    "impact": "positive" if v_status else "neutral"
                },
                {
                    "label": "Localisation",
                    "value": f"Zone {data.get('state')}",
                    "impact": "positive" if v_rate > 0.65 else "negative"
                }
            ],
            "forensic_insights": {
                "vulnerability_score": "ÉLEVÉ" if v_status else "MODÉRÉ",
                "risk_profile": "Cas Prioritaire" if prob > 0.75 else (
                    "Cas Standard" if prob > 0.4 else "Cold Case Potentiel"
                ),
                "investigation_advice": (
                    "Focus sur preuves balistiques" if case_df['Weapon_Category'][0] == 'Firearm'
                    else "Recherche de témoins"
                )
            },
            "analysis": (
                f"Probabilité de résolution : {prob:.1%}. "
                f"Profil victime ({v_label}) en {data.get('state')} suggère priorité "
                f"{'ÉLEVÉE' if v_status or prob > 0.7 else 'NORMALE'}. "
                f"{'Analyse balistique recommandée.' if case_df['Weapon_Category'][0] == 'Firearm' else ''}"
            )
        })

    except Exception as e:
        print(f"❌ Erreur : {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n🚀 Serveur démarré sur http://localhost:5000")
    print("📊 Interface Web : http://localhost:5000")
    print("=" * 60)
    app.run(port=5000, debug=True)
