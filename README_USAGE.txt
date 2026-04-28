🚀 GUIDE D'UTILISATION - SYSTÈME DE PRÉDICTION D'HOMICIDES
================================================================

## 📋 Vue d'ensemble

Système complet de prédiction de solvabilité d'homicides utilisant :
- ✅ Interface Web moderne (crime_predictor.html)
- ✅ Backend Flask (app.py) 
- ✅ Modèle Machine Learning (Random Forest)
- ✅ Analyse forensique avancée

---

## 🔧 INSTALLATION & DÉMARRAGE

### 1️⃣ Vérifier les dépendances

```bash
pip install flask flask-cors pandas joblib scikit-learn
```

### 2️⃣ Démarrer le serveur

```bash
cd c:\Users\DELL\Desktop\Downloads\homicide_data_mining
python app.py
```

Vous devriez voir :
```
============================================================
DÉMARRAGE DU SERVEUR IA
============================================================
✅ Modèle chargé : [nombre] features
🚀 Serveur démarré sur http://localhost:5000
📊 Interface Web : http://localhost:5000
```

### 3️⃣ Accéder à l'interface

Ouvrez votre navigateur à : **http://localhost:5000**

---

## 📊 UTILISATION DE L'INTERFACE

### Formulaire à remplir :

1. **Sexe de la victime** : Male / Female / Unknown
2. **Âge de la victime** : 0-100 ans
3. **Race** : Black / White / Hispanic / Asian / Unknown
4. **Nombre de victimes** : Entier ≥ 1
5. **Année** : 1980-2025
6. **Type d'arme** : Firearm / Knife / Blunt Object / Strangulation / Unknown
7. **État américain** : Sélectionner dans la liste

### Résultats affichés :

✅ **Probabilité de résolution** : 0-100% (en grande police)
✅ **Verdict** : "Probablement Résoluble" ou "Risque de Cold Case"
✅ **Profil de risque** : Cas Prioritaire / Standard / Cold Case Potentiel
✅ **Niveau de vulnérabilité** : Élevé ou Modéré
✅ **Conseil d'enquête** : Preuves balistiques ou Recherche de témoins
✅ **Analyse détaillée** : Rapport synthétique complet

---

## 🤖 ENDPOINTS API

### GET /api/features
Retourne les features du modèle.
```bash
curl http://localhost:5000/api/features
```

Réponse :
```json
{
  "features_count": 42,
  "features": ["Victim_Sex_Female", ...],
  "model_status": "ready"
}
```

### POST /api/predict
Prédiction pour un cas.
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sex": "Male",
    "age": 35,
    "race": "White",
    "weapon": "Firearm",
    "state": "New York",
    "year": 2024,
    "victims": 1,
    "historicalRate": 0.54
  }'
```

Réponse :
```json
{
  "probability": 0.75,
  "probability_percent": "75%",
  "verdict": "Resolved",
  "vulnerability": false,
  "forensic_insights": {
    "vulnerability_score": "MODÉRÉ",
    "risk_profile": "Cas Prioritaire",
    "investigation_advice": "Focus sur preuves balistiques"
  },
  "analysis": "Analyse détaillée..."
}
```

---

## 📁 STRUCTURE DES FICHIERS

```
homicide_data_mining/
├── app.py                          ← Backend Flask (API)
├── crime_predictor.html            ← Interface Web
├── homicide_data_mining/
│   ├── database.csv                ← Données brutes (~112 MB)
│   ├── homicide_model_optimized.pkl   ← Modèle ML (3.2 MB)
│   ├── model_features_optimized.json  ← Features (669 B)
│   ├── test.py                     ← Script de test
│   ├── criminal_profiling.py       ← Analyse victimologique
│   └── Construire le Modèle Random Forest.py ← Construction du modèle
├── README_USAGE.txt                ← Ce fichier
└── NETTOYAGE_RAPPORT.txt           ← Historique nettoyage
```

---

## 🔬 FICHIERS PYTHON DISPONIBLES

### test.py
Teste et évalue le modèle avec données réelles.
```bash
python homicide_data_mining/test.py
```

### criminal_profiling.py
Analyse victimologique et profilage criminel.
```bash
python homicide_data_mining/criminal_profiling.py
```

### Construire le Modèle Random Forest.py
Reconstruit le modèle depuis les données brutes.
```bash
python "homicide_data_mining/Construire le Modèle Random Forest.py"
```

---

## 📊 PERFORMANCE DU MODÈLE

- **Type** : Random Forest Classifier
- **Arbres** : 100
- **AUC-ROC** : ~0.82
- **Accuracy** : ~75%
- **Features** : 42 variables encoder après one-hot encoding
- **Données d'entraînement** : ~20,000+ cas résolus/non-résolus

---

## 🔐 SÉCURITÉ & LIMITATIONS

⚠️ **Important** :
- Modèle pour analyse exploratoire uniquement
- Pas de causalité, seulement corrélations historiques
- Les prédictions dépendent de la qualité des données d'entrée
- Les taux historiques par État sont approximatifs

---

## ⚙️ DÉPANNAGE

### ❌ "Serveur IA non accessible"
**Solution** : Assurez-vous que :
1. Le serveur est lancé : `python app.py`
2. Aucun firewall ne bloque le port 5000
3. Le modèle est bien présent : `homicide_data_mining/homicide_model_optimized.pkl`

### ❌ "Modèle non chargé"
**Solution** : Vérifiez que :
1. `homicide_model_optimized.pkl` existe
2. `model_features_optimized.json` existe
3. Les chemins sont corrects dans app.py

### ❌ "Données numériques invalides"
**Solution** : Vérifiez que :
1. L'âge est entre 0 et 100
2. L'année est entre 1980 et 2025
3. Le nombre de victimes est ≥ 1

---

## 📈 EXEMPLE D'UTILISATION COMPLÈTE

1. Ouvrez un terminal
2. Naviguez vers le dossier projet
3. Lancez le serveur :
   ```bash
   python app.py
   ```
4. Ouvrez : http://localhost:5000
5. Remplissez le formulaire
6. Cliquez "Lancer l'Analyse"
7. Attendez la prédiction (~1-2 secondes)
8. Lisez l'analyse forensique complète

---

## 🎯 CAS D'USAGE RECOMMANDÉS

✅ **Bons cas** :
- Cas avec informations complètes (sexe, âge, arme connue)
- Crimes dans des États avec données historiques complètes
- Analyses exploratoires et comparatives

❌ **Cas problématiques** :
- Données très incomplètes
- États avec peu de données historiques
- Signalement de "Unknown" pour presque tous les champs

---

## 📞 SUPPORT TECHNIQUE

En cas de problème :
1. Vérifiez la console du serveur pour les erreurs
2. Vérifiez la console du navigateur (F12 > Console)
3. Assurez-vous que tous les fichiers essentiels existent
4. Relancez le serveur

---

**Version** : 2.0 (Avril 2026)
**Modèle** : Random Forest Classifier v1
**État** : ✅ Production Ready
**Dernière mise à jour** : 26/04/2026
