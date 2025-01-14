import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Import local modules
from src.spatial_analysis import SpatialAnalysis

class RealEstateModeling:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.feature_importance = {}
        # La méthode d'analyse spatiale est directement ajoutée ici
        self.analyze_spatial_predictions = lambda quartiers: self._analyze_spatial_predictions(quartiers)
        
        # Création des dossiers pour les résultats
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_path = os.path.join(self.base_path, 'results')
        self.viz_path = os.path.join(self.results_path, 'visualizations')
        self.metrics_path = os.path.join(self.results_path, 'metrics')
        self.spatial_path = os.path.join(self.results_path, 'spatial')
        
        # Création des dossiers
        for path in [self.results_path, self.viz_path, self.metrics_path, self.spatial_path]:
            os.makedirs(path, exist_ok=True)
        
        # Création des sous-dossiers pour les visualisations
        for subdir in ['model_comparison', 'linear_regression', 'xgboost']:
            os.makedirs(os.path.join(self.viz_path, subdir), exist_ok=True)

    def save_visualization(self, fig, filename: str, subdir: str = None) -> None:
        """Save visualization to the appropriate directory"""
        if subdir:
            save_path = os.path.join(self.viz_path, subdir)
        else:
            save_path = self.viz_path
        
        fig.savefig(os.path.join(save_path, filename), bbox_inches='tight', dpi=300)
        plt.close(fig)

    def save_metrics(self) -> None:
        """Save metrics to JSON files"""
        for model_name, metrics in self.metrics.items():
            filename = f"{model_name}_metrics.json"
            with open(os.path.join(self.metrics_path, filename), 'w') as f:
                json.dump(metrics, f, indent=4)

    def _analyze_spatial_predictions(self, quartiers) -> None:
        """Analyze predictions spatially"""
        coordinates = pd.DataFrame({
            'x': self.X_test['x'],
            'y': self.X_test['y']
        })
        
        spatial_results = {}
        
        for model_name, predictions in self.predictions.items():
            print(f"\nAnalyse spatiale pour le modèle {model_name}...")
            
            analyzer = SpatialAnalysis(
                predictions['test'],
                self.y_test,
                coordinates,
                quartiers
            )
            
            # Calcul et sauvegarde des statistiques spatiales
            stats = analyzer.calculate_spatial_errors()
            stats.to_csv(os.path.join(self.spatial_path, f"{model_name}_spatial_stats.csv"))
            print(f"\nStatistiques par quartier - {model_name}:")
            print(stats)
            
            # Création et sauvegarde des visualisations
            print("\nCréation des visualisations...")
            analyzer.plot_errors_by_quartier(self.viz_path, model_name)
            analyzer.plot_price_comparison(self.viz_path, model_name)

    def train_linear_regression(self) -> None:
        """Train a multiple linear regression model"""
        print("\nEntraînement de la régression linéaire...")
        lr_model = LinearRegression()
        
        # Cross-validation
        cv_scores = cross_validate(lr_model, self.X_train, self.y_train,
                                 cv=5,
                                 scoring=['r2', 'neg_mean_squared_error'],
                                 return_train_score=True)
        
        print("Scores de cross-validation:")
        print(f"R² CV: {cv_scores['test_r2'].mean():.3f} (+/- {cv_scores['test_r2'].std() * 2:.3f})")
        print(f"RMSE CV: {np.sqrt(-cv_scores['test_neg_mean_squared_error'].mean()):.0f}")
        
        # Entraînement final
        lr_model.fit(self.X_train, self.y_train)
        
        # Store model and make predictions
        self.models['linear'] = lr_model
        self.predictions['linear'] = {
            'train': lr_model.predict(self.X_train),
            'test': lr_model.predict(self.X_test)
        }
        
        # Calculate feature importance (coefficients)
        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': np.abs(lr_model.coef_)
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['linear'] = importance

    def train_xgboost(self, tune_hyperparameters: bool = True) -> None:
        """Train XGBoost model with hyperparameter tuning"""
        print("\nEntraînement du modèle XGBoost...")
        
        if tune_hyperparameters:
            print("Optimisation des hyperparamètres...")
            param_grid = {
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0]
            }
            
            best_params = self._tune_xgboost_params(param_grid)
            final_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
                **best_params
            )
        else:
            final_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                n_jobs=-1
            )
        
        # Entraînement final
        final_model.fit(self.X_train, self.y_train)
        
        self.models['xgboost'] = final_model
        self.predictions['xgboost'] = {
            'train': final_model.predict(self.X_train),
            'test': final_model.predict(self.X_test)
        }
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['xgboost'] = importance

    def _tune_xgboost_params(self, param_grid: Dict) -> Dict:
        """Helper method for XGBoost parameter tuning"""
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        best_score = float('inf')
        best_params = None
        
        param_combinations = [dict(zip(param_grid.keys(), v)) for v 
                           in product(*param_grid.values())]
        
        print(f"Test de {len(param_combinations)} combinaisons de paramètres...")
        
        for params in param_combinations:
            cv_scores = []
            for train_idx, val_idx in kf.split(self.X_train):
                X_train_cv = self.X_train.iloc[train_idx]
                y_train_cv = self.y_train.iloc[train_idx]
                X_val_cv = self.X_train.iloc[val_idx]
                y_val_cv = self.y_train.iloc[val_idx]
                
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    random_state=42,
                    n_jobs=-1,
                    **params
                )
                
                model.fit(X_train_cv, y_train_cv)
                pred = model.predict(X_val_cv)
                rmse = np.sqrt(mean_squared_error(y_val_cv, pred))
                cv_scores.append(rmse)
            
            mean_score = np.mean(cv_scores)
            
            if mean_score < best_score:
                best_score = mean_score
                best_params = params
                print(f"\nNouvelle meilleure RMSE: {best_score:,.0f} €")
                print("Paramètres:", best_params)
        
        return best_params

    def calculate_metrics(self) -> None:
        """Calculate performance metrics for all models"""
        for model_name in self.predictions.keys():
            train_pred = self.predictions[model_name]['train']
            test_pred = self.predictions[model_name]['test']
            
            self.metrics[model_name] = {
                'train': {
                    'rmse': float(np.sqrt(mean_squared_error(self.y_train, train_pred))),
                    'mae': float(mean_absolute_error(self.y_train, train_pred)),
                    'r2': float(r2_score(self.y_train, train_pred))
                },
                'test': {
                    'rmse': float(np.sqrt(mean_squared_error(self.y_test, test_pred))),
                    'mae': float(mean_absolute_error(self.y_test, test_pred)),
                    'r2': float(r2_score(self.y_test, test_pred))
                }
            }
        
        # Sauvegarder les métriques
        self.save_metrics()

    def plot_predictions(self) -> None:
        """Plot actual vs predicted values for all models"""
        n_models = len(self.predictions)
        fig, axes = plt.subplots(n_models, 2, figsize=(15, 5*n_models))
        
        for idx, (model_name, preds) in enumerate(self.predictions.items()):
            # Training set
            axes[idx, 0].scatter(self.y_train, preds['train'], alpha=0.5)
            axes[idx, 0].plot([self.y_train.min(), self.y_train.max()],
                            [self.y_train.min(), self.y_train.max()],
                            'r--', lw=2)
            axes[idx, 0].set_title(f'{model_name} - Training Set')
            axes[idx, 0].set_xlabel('Valeurs réelles (€)')
            axes[idx, 0].set_ylabel('Valeurs prédites (€)')
            
            # Test set
            axes[idx, 1].scatter(self.y_test, preds['test'], alpha=0.5)
            axes[idx, 1].plot([self.y_test.min(), self.y_test.max()],
                            [self.y_test.min(), self.y_test.max()],
                            'r--', lw=2)
            axes[idx, 1].set_title(f'{model_name} - Test Set')
            axes[idx, 1].set_xlabel('Valeurs réelles (€)')
            axes[idx, 1].set_ylabel('Valeurs prédites (€)')
        
        plt.tight_layout()
        self.save_visualization(fig, 'predictions_scatter.png', 'model_comparison')

    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot feature importance for all models"""
        n_models = len(self.feature_importance)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 6*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importance) in enumerate(self.feature_importance.items()):
            top_features = importance.head(top_n)
            
            sns.barplot(data=top_features, x='importance', y='feature', ax=axes[idx])
            axes[idx].set_title(f'Top {top_n} variables importantes - {model_name}')
            axes[idx].set_xlabel('Importance')
            axes[idx].set_ylabel('Variable')
        
        plt.tight_layout()
        self.save_visualization(fig, 'feature_importance.png', 'model_comparison')

def main():
    """Main function to run the modeling pipeline"""
    # Get preprocessed data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    preprocessor = RealEstatePreprocessor(data_path)
    
    # Récupération des données avec les quartiers
    X_train, X_test, y_train, y_test, _, quartiers_test = preprocessor.main()
    
    # Initialize modeling class
    modeling = RealEstateModeling(X_train, X_test, y_train, y_test)
    
    # Train models
    print("\nEntraînement des modèles...")
    modeling.train_linear_regression()
    modeling.train_xgboost(tune_hyperparameters=True)
    
    # Calculate and display metrics
    modeling.calculate_metrics()
    print("\nMétriques de performance des modèles:")
    for model_name, metrics in modeling.metrics.items():
        print(f"\n{model_name.upper()}:")
        print("Métriques d'entraînement:")
        print(f"RMSE: {metrics['train']['rmse']:,.0f} €")
        print(f"MAE: {metrics['train']['mae']:,.0f} €")
        print(f"R²: {metrics['train']['r2']:.3f}")
        print("\nMétriques de test:")
        print(f"RMSE: {metrics['test']['rmse']:,.0f} €")
        print(f"MAE: {metrics['test']['mae']:,.0f} €")
        print(f"R²: {metrics['test']['r2']:.3f}")
    
    # Plot results
    print("\nCréation des visualisations de performance...")
    modeling.plot_predictions()
    modeling.plot_feature_importance()
    
    # Analyse spatiale avec les quartiers
    print("\nAnalyse spatiale des prédictions...")
    print(f"Nombre de quartiers dans le jeu de test : {quartiers_test.nunique()}")
    modeling.analyze_spatial_predictions(quartiers_test)

if __name__ == "__main__":
    from itertools import product
    from src.real_estate_preprocessor import RealEstatePreprocessor
    
    # Obtenir les données
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    preprocessor = RealEstatePreprocessor(data_path)
    X_train, X_test, y_train, y_test, _, quartiers_test = preprocessor.main()
    
    # Initialiser le modeling
    modeling = RealEstateModeling(X_train, X_test, y_train, y_test)
    
    # Entraîner les modèles
    print("\nEntraînement des modèles...")
    modeling.train_linear_regression()
    modeling.train_xgboost(tune_hyperparameters=True)
    
    # Créer et entrainer le preprocesseur
    from models.model_utils import create_preprocessor, save_model_and_preprocessor
    preprocessor = create_preprocessor(X_train)
    preprocessor.fit(X_train)
    
    # Prétraiter les données
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Sauvegarder le meilleur modèle et le preprocesseur
    best_model = modeling.models['xgboost']
    base_path = os.path.dirname(os.path.dirname(__file__))
    save_model_and_preprocessor(best_model, preprocessor, base_path)
    
    # Calculer et afficher les métriques
    modeling.calculate_metrics()
    print("\nMétriques de performance des modèles:")
    for model_name, metrics in modeling.metrics.items():
        print(f"\n{model_name.upper()}:")
        for set_name in ['train', 'test']:
            print(f"\nMétriques de {set_name}:")
            print(f"RMSE: {metrics[set_name]['rmse']:,.0f} €")
            print(f"MAE: {metrics[set_name]['mae']:,.0f} €")
            print(f"R²: {metrics[set_name]['r2']:.3f}")
    
    # Créer les visualisations
    print("\nCréation des visualisations...")
    modeling.plot_predictions()
    modeling.plot_feature_importance()
    
    # Analyse spatiale
    print("\nAnalyse spatiale des prédictions...")
    print(f"Nombre de quartiers dans le jeu de test : {quartiers_test.nunique()}")
    modeling.analyze_spatial_predictions(quartiers_test)