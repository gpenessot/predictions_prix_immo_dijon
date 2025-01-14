import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def create_preprocessor(X_train):
    """Create and fit the preprocessor pipeline"""
    # Identifier les colonnes numériques
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Créer le preprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def save_model_and_preprocessor(model, preprocessor, base_path):
    """Save model and preprocessor to disk"""
    # Créer le dossier models s'il n'existe pas
    models_dir = os.path.join(base_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = os.path.join(models_dir, 'xgboost_model.joblib')
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé: {model_path}")
    
    # Sauvegarder le preprocesseur
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocesseur sauvegardé: {preprocessor_path}")