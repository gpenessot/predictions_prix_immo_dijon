import os
import glob
import json
from typing import Dict, List, Tuple, Any

import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class RealEstatePreprocessor:
    def __init__(self, data_path: str):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the raw data directory
        """
        self.data_path = data_path
        self.data_type = ['tram', 'bus', 'velo', 'commerces', 'immo', 'ecoles']
        self.input_data = None
        self.df_immo = None
        self.df_bus = None
        self.df_tram = None
        self.df_ecoles = None
        self.df_commerces = None
        self.quartiers = None
        
        # Création des dossiers pour les données traitées et les statistiques
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.processed_path = os.path.join(self.base_path, 'data', 'processed')
        self.stats_path = os.path.join(self.base_path, 'results', 'preprocessing_stats')
        
        for path in [self.processed_path, self.stats_path]:
            os.makedirs(path, exist_ok=True)
    
    def save_preprocessing_stats(self, stats: Dict) -> None:
        """
        Save preprocessing statistics to JSON
        
        Args:
            stats: Dictionary containing preprocessing statistics
        """
        with open(os.path.join(self.stats_path, 'preprocessing_stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)

    def clean_data(self) -> Dict:
        """
        Clean and transform the real estate data
        
        Returns:
            Dictionary containing preprocessing statistics
        """
        # Statistiques initiales
        n_initial = len(self.df_immo)
        stats = {'initial_count': n_initial}
        
        print("\nÉtape 1: Suppression des biens atypiques...")
        
        # Filtre sur la taille des terrains
        self.df_immo = self.df_immo[self.df_immo['sterr'] <= 10000]
        stats['after_terrain_filter'] = len(self.df_immo)
        
        # Calculer les quantiles pour prix_m2 et valeurfonc
        q_prix = self.df_immo['prix_m2'].quantile([0.01, 0.99])
        q_valeur = self.df_immo['valeurfonc'].quantile([0.01, 0.99])
        
        stats['price_thresholds'] = {
            'prix_m2_min': float(q_prix[0.01]),
            'prix_m2_max': float(q_prix[0.99]),
            'valeur_min': float(q_valeur[0.01]),
            'valeur_max': float(q_valeur[0.99])
        }
        
        # Filtrer les valeurs extrêmes
        self.df_immo = self.df_immo[
            (self.df_immo['prix_m2'] >= q_prix[0.01]) & 
            (self.df_immo['prix_m2'] <= q_prix[0.99]) &
            (self.df_immo['valeurfonc'] >= q_valeur[0.01]) & 
            (self.df_immo['valeurfonc'] <= q_valeur[0.99])
        ]
        
        n_final = len(self.df_immo)
        stats['final_count'] = n_final
        stats['removed_count'] = n_initial - n_final
        stats['removed_percentage'] = round(((n_initial - n_final)/n_initial)*100, 1)
        
        print(f"Nombre de biens filtrés : {stats['removed_count']} ({stats['removed_percentage']}%)")
        print(f"Seuils retenus:")
        print(f"- Prix/m² : entre {q_prix[0.01]:,.0f}€ et {q_prix[0.99]:,.0f}€")
        print(f"- Valeur foncière : entre {q_valeur[0.01]:,.0f}€ et {q_valeur[0.99]:,.0f}€")

        print("\nÉtape 2: Standardisation des types de propriété...")
        typo_mapping = {
            '4 pièces': 'T4', 
            '5 pièces ou plus': 'T5+', 
            'T5 ou plus': 'T5+',
            '3 pièces': 'T3',
            '2 pièces': 'T2',
            '1 pièce': 'T1'
        }
        self.df_immo['typologie'] = self.df_immo['typologie'].replace(typo_mapping)
        
        print("\nÉtape 3: Conversion des types de données...")
        self.df_immo['anneemut'] = pd.to_numeric(self.df_immo['anneemut'], errors='coerce')
        self.df_immo['moismut'] = pd.to_numeric(self.df_immo['moismut'], errors='coerce')
        
        print("\nÉtape 4: Sauvegarde des quartiers...")
        self.quartiers = self.df_immo['quartier'].copy()
        stats['quartiers_count'] = self.quartiers.nunique()
        stats['quartiers_distribution'] = self.quartiers.value_counts().to_dict()
        
        print("\nÉtape 5: Suppression des colonnes inutiles...")
        cols_to_drop = [
            'libnatmut', 'l_codinsee', 'quartier', 'code_quartier',
            'nom_iris', 'code_iris', 'code_insee', 'datemut'
        ]
        self.df_immo = self.df_immo.drop(labels=cols_to_drop, axis=1)
        
        print("\nÉtape 6: Extraction des coordonnées...")
        df_temp = self.df_immo.copy()
        df_temp = df_temp.to_crs(epsg=4326)
        self.df_immo[['x', 'y']] = df_temp.get_coordinates(ignore_index=True)
        self.df_immo = self.df_immo.drop(labels=['geo_point_2d', 'geometry'], axis=1)
        
        print("\nÉtape 7: Gestion des valeurs manquantes...")
        # Affichage des colonnes avec des valeurs manquantes
        nan_cols = self.df_immo.isna().sum()
        stats['missing_values'] = nan_cols[nan_cols > 0].to_dict()
        
        print("\nColonnes avec valeurs manquantes avant imputation:")
        print(nan_cols[nan_cols > 0])
        
        # Imputation des valeurs numériques manquantes
        numeric_cols = self.df_immo.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            self.df_immo[numeric_cols] = imputer.fit_transform(self.df_immo[numeric_cols])
        
        # Imputation des valeurs catégorielles
        categorical_cols = self.df_immo.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            self.df_immo[categorical_cols] = imputer.fit_transform(self.df_immo[categorical_cols])
        
        stats['final_dtypes'] = self.df_immo.dtypes.astype(str).to_dict()
        
        print("\nVérification des types de données:")
        print(self.df_immo.dtypes)
        
        return stats

    def load_data(self) -> None:
        """Load all necessary geojson files and project them to a local CRS"""
        input_data = glob.glob(os.path.join(self.data_path, '*.geojson'))
        input_data.sort()
        self.input_data = {i:j for i,j in zip(self.data_type, input_data)}
        
        try:
            self.df_immo = gpd.read_file(self.input_data['immo'])
            self.df_bus = gpd.read_file(self.input_data['bus'])
            self.df_tram = gpd.read_file(self.input_data['tram'])
            self.df_ecoles = gpd.read_file(self.input_data['ecoles'])
            self.df_commerces = gpd.read_file(self.input_data['commerces'])
            
            # Projection en Lambert 93
            self.df_immo = self.df_immo.to_crs(epsg=2154)
            self.df_bus = self.df_bus.to_crs(epsg=2154)
            self.df_tram = self.df_tram.to_crs(epsg=2154)
            self.df_ecoles = self.df_ecoles.to_crs(epsg=2154)
            self.df_commerces = self.df_commerces.to_crs(epsg=2154)
            
        except Exception as e:
            raise Exception(f"Error loading data files: {str(e)}")

    def calculate_nearby_shops(self, radius_meters: float = 150) -> pd.Series:
        """
        Calculate number of shops within specified radius for each property
        
        Args:
            radius_meters: Search radius in meters
            
        Returns:
            Series containing shop counts
        """
        buffer_deg = radius_meters  # Plus besoin de conversion car on est en Lambert 93
        df_immo_buffer = self.df_immo.buffer(buffer_deg)
        
        commerces_proches = gpd.sjoin(
            self.df_commerces, 
            gpd.GeoDataFrame(geometry=gpd.GeoSeries(df_immo_buffer), crs=self.df_immo.crs), 
            how="inner", 
            predicate="intersects"
        )
        
        commerces_count = commerces_proches.groupby('index_right')['nom_enseigne'].count()
        full_index = pd.Index(range(len(self.df_immo)))
        commerces_count = commerces_count.reindex(full_index, fill_value=0)
        
        return commerces_count

    def calculate_min_distances(self) -> None:
        """Calculate minimum distances to various amenities"""
        for df_name, df in [('bus', self.df_bus), ('tram', self.df_tram), ('ecole', self.df_ecoles)]:
            self.df_immo[f'min_{df_name}_id'] = self.df_immo.geometry.apply(
                lambda g: df.distance(g).idxmin()
            )
            self.df_immo[f'min_{df_name}_dist'] = self.df_immo.geometry.apply(
                lambda g: df.distance(g).min()
            )

    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training
        
        Returns:
            Tuple containing features DataFrame and target Series
        """
        df_encoded = self.encode_categorical_features()
        
        # Retirer prix_m2 des features
        if 'prix_m2' in df_encoded.columns:
            print("\nSuppression de la variable prix_m2 des features...")
            df_encoded = df_encoded.drop('prix_m2', axis=1)
        
        # Separate features and target
        y = df_encoded['valeurfonc']
        X = df_encoded.drop('valeurfonc', axis=1)
        
        # Save processed features
        X.to_csv(os.path.join(self.processed_path, 'features.csv'), index=False)
        y.to_csv(os.path.join(self.processed_path, 'target.csv'), index=False)
        
        print("\nFeatures utilisées pour la prédiction:")
        print(X.columns.tolist())
        
        return X, y

    def encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical features using Label and One-Hot encoding
        
        Returns:
            DataFrame with encoded features
        """
        df_encoded = self.df_immo.copy()
        
        # Label encode type_bien
        le = LabelEncoder()
        df_encoded["type_bien"] = le.fit_transform(df_encoded["type_bien"].astype(str))
        
        # One-hot encode commune and typologie
        for col in ['commune', 'typologie']:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            feature_array = ohe.fit_transform(df_encoded[col].astype(str).values.reshape(-1, 1))
            feature_labels = [f"{col}_{str(label)}" for label in ohe.get_feature_names_out()]
            
            df_encoded = pd.concat([
                df_encoded.drop(col, axis=1),
                pd.DataFrame(feature_array, columns=feature_labels, index=df_encoded.index)
            ], axis=1)
        
        return df_encoded

    def main(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
            """
            Run the complete preprocessing pipeline
            
            Returns:
                Tuple containing (X_train, X_test, y_train, y_test, quartiers_train, quartiers_test)
            """
            print("\nDébut du prétraitement...")
            
            print("\nChargement des données...")
            self.load_data()
            
            print("\nCalcul des commerces à proximité...")
            commerces_proches = self.calculate_nearby_shops()
            self.df_immo['nb_commerces_150m'] = commerces_proches
            
            print("\nCalcul des distances minimales...")
            self.calculate_min_distances()
            
            print("\nNettoyage des données...")
            stats = self.clean_data()
            self.save_preprocessing_stats(stats)
            
            print("\nPréparation des features...")
            X, y = self.prepare_features()
            
            print("\nSplit des données...")
            # Création d'un index pour le split
            indices = np.arange(len(X))
            X_train_idx, X_test_idx, y_train, y_test = train_test_split(
                indices, y, test_size=0.2, random_state=42
            )
            
            # Split des features et des quartiers en utilisant les mêmes indices
            X_train = X.iloc[X_train_idx]
            X_test = X.iloc[X_test_idx]
            quartiers_train = self.quartiers.iloc[X_train_idx]
            quartiers_test = self.quartiers.iloc[X_test_idx]
            
            # Sauvegarde des splits
            split_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'quartiers_train': quartiers_train,
                'quartiers_test': quartiers_test
            }
            
            # Sauvegarde des données splittées
            for name, data in split_data.items():
                data.to_csv(os.path.join(self.processed_path, f'{name}.csv'), index=False)
            
            print("\nPrétraitement terminé !")
            print(f"\nNombre de features : {X.shape[1]}")
            print(f"Nombre d'observations : {X.shape[0]}")
            print(f"Nombre de quartiers uniques : {self.quartiers.nunique()}")
            
            return X_train, X_test, y_train, y_test, quartiers_train, quartiers_test

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    preprocessor = RealEstatePreprocessor(data_path)
    X_train, X_test, y_train, y_test, quartiers_train, quartiers_test = preprocessor.main()
    print("\nDimensions finales :")
    print(f"Training set : {X_train.shape}")
    print(f"Test set : {X_test.shape}")