@echo off
REM Script de démarrage du serveur IA - Homicide Data Mining
REM ======================================================

TITLE Crime Predictor - AI Server

echo.
echo ======================================================
echo   CRIME PREDICTOR - Serveur IA de Prediction
echo ======================================================
echo.
echo Verification des dependances...
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe!
    echo Installez Python depuis https://www.python.org/downloads/
    pause
    exit /b 1
)

echo OK: Python detecte!
echo.

REM Vérifier les dépendances
echo Verification de Flask...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installation de Flask...
    pip install flask flask-cors pandas joblib scikit-learn
    echo.
)

REM Vérifier que les fichiers essentiels existent
echo Verification des fichiers essentiels...
if not exist "homicide_data_mining\homicide_model_optimized.pkl" (
    echo ERREUR: Modele non trouve!
    echo Fichier manquant: homicide_data_mining\homicide_model_optimized.pkl
    pause
    exit /b 1
)

if not exist "homicide_data_mining\model_features_optimized.json" (
    echo ERREUR: Features non trouvees!
    echo Fichier manquant: homicide_data_mining\model_features_optimized.json
    pause
    exit /b 1
)

if not exist "crime_predictor.html" (
    echo ERREUR: Interface Web non trouvee!
    echo Fichier manquant: crime_predictor.html
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ERREUR: Serveur Python non trouve!
    echo Fichier manquant: app.py
    pause
    exit /b 1
)

echo OK: Tous les fichiers sont presents!
echo.

REM Démarrer le serveur
echo ======================================================
echo Demarrage du serveur...
echo ======================================================
echo.

python app.py

REM Si le serveur se ferme
echo.
echo Le serveur s'est arrête.
pause
