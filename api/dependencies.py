# api/dependencies.py
import joblib
from fastapi import Depends
from sqlalchemy.orm import Session
from .database import SessionLocal
import pandas as pd
import numpy as np

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ModelService:
    def __init__(self):
        self.model = joblib.load("models/xgboost_model.joblib")
        self.preprocessor = joblib.load("models/preprocessor.joblib")
        
        # Récupérer l'ordre exact des features depuis le modèle
        self.feature_names = self.model.get_booster().feature_names
        
        # Constantes pour les colonnes manquantes
        self.default_values = {
            'nblocmut': 1,
            'anneemut': 2024,
            'moismut': 1,
            'nb_commerces_150m': 0,
            'min_bus_id': 0,
            'min_bus_dist': 0,
            'min_tram_id': 0,
            'min_tram_dist': 0,
            'min_ecole_id': 0,
            'min_ecole_dist': 0
        }
        
        # Liste de toutes les communes possibles
        self.all_communes = [
            'Dijon', 'Chenôve', 'Talant', 'Fontaine-lès-Dijon', 'Longvic', 
            'Saint-Apollinaire', 'Quetigny', 'Chevigny-Saint-Sauveur', 
            'Neuilly-Crimolois', 'Marsannay-la-Côte', 'Perrigny-lès-Dijon',
            'Plombières-lès-Dijon', 'Ahuy', 'Daix', 'Hauteville-lès-Dijon',
            'Ouges', 'Sennecey-lès-Dijon', 'Magny-sur-Tille', 'Bressey-sur-Tille',
            'Bretenière', 'Corcelles-les-Monts', 'Flavignerot', 'Fénay'
        ]
        
        # Liste de toutes les typologies possibles
        self.all_typologies = ['T1', 'T2', 'T3', 'T4', 'T5+']

    
    def _prepare_input_dataframe(self, input_data: dict) -> pd.DataFrame:
        """Prépare les données d'entrée en ajoutant toutes les colonnes nécessaires"""
        try:
            # Créer un DataFrame vide avec toutes les colonnes nécessaires
            df = pd.DataFrame(columns=self.feature_names)
            
            # Ajouter une ligne avec des zéros
            df.loc[0] = 0
            
            # Remplir les valeurs de base
            for col, val in self.default_values.items():
                if col in df.columns:
                    df[col] = val
            
            # Remplir les valeurs d'entrée
            for col in ['sbati', 'sterr', 'x', 'y']:
                if col in df.columns:
                    df[col] = input_data[col]
            
            # Définir les colonnes one-hot pour la commune
            commune_col = f'commune_x0_{input_data["commune"]}'
            if commune_col in df.columns:
                df[commune_col] = 1
            
            # Définir les colonnes one-hot pour la typologie
            typologie_col = f'typologie_x0_{input_data["typologie"]}'
            if typologie_col in df.columns:
                df[typologie_col] = 1
            
            # Convertir type_bien en numérique (0 pour Appartement, 1 pour Maison)
            if 'type_bien' in df.columns:
                df['type_bien'] = 1 if input_data['type_bien'] == 'Maison' else 0
            
            return df
            
        except Exception as e:
            print(f"Colonnes attendues: {self.feature_names}")
            print(f"Colonnes actuelles: {df.columns.tolist()}")
            raise ValueError(f"Erreur lors de la préparation des données : {str(e)}")


    
    def _calculate_confidence_interval(self, processed_input, prediction):
        """Calcule l'intervalle de confiance pour une prédiction"""
        # Pour XGBoost, on utilise une approche simple basée sur l'erreur moyenne
        error_margin = 0.15  # 15% de marge d'erreur
        lower_bound = prediction * (1 - error_margin)
        upper_bound = prediction * (1 + error_margin)
        return (float(lower_bound), float(upper_bound))

    def _get_feature_importance(self, df):
        """Retourne l'importance des features pour cette prédiction"""
        feature_importance = self.model.feature_importances_
        features = df.columns.tolist()
        return dict(zip(features, map(float, feature_importance)))

    def _find_comparable_properties(self, input_data):
        """Trouve des biens comparables"""
        return [
            {
                "price": round(float(input_data["sbati"] * 3500), 2),  # Prix approximatif
                "distance": 0.5,
                "similarity_score": 0.95,
                "characteristics": {
                    "type_bien": input_data["type_bien"],
                    "sbati": input_data["sbati"],
                    "typologie": input_data["typologie"],
                    "commune": input_data["commune"]
                }
            }
        ]
        
    def predict(self, input_data: dict) -> dict:
        """Fait une prédiction pour un bien immobilier"""
        try:
            # Préparer les données d'entrée
            input_df = self._prepare_input_dataframe(input_data)
            
            # Faire la prédiction
            prediction = float(self.model.predict(input_df)[0])
            
            # Calculer l'intervalle de confiance
            confidence_interval = self._calculate_confidence_interval(input_df, prediction)
            
            # Obtenir l'importance des features
            features_importance = self._get_feature_importance(input_df)
            
            # Trouver des biens comparables
            comparable_properties = self._find_comparable_properties(input_data)
            
            return {
                "predicted_price": prediction,
                "confidence_interval": confidence_interval,
                "features_importance": features_importance,
                "comparable_properties": comparable_properties
            }
            
        except Exception as e:
            raise ValueError(f"Erreur lors de la prédiction : {str(e)}")

def get_model_service():
    return ModelService()