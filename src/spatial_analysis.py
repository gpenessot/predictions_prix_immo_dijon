import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SpatialAnalysis:
    def __init__(self, predictions, actual_values, coordinates, quartiers):
        """
        Initialize the spatial analysis
        
        Args:
            predictions: Array of predicted values
            actual_values: Array of actual values
            coordinates: DataFrame with x, y coordinates
            quartiers: Series of quartier values
        """
        self.predictions = predictions
        self.actual_values = actual_values
        self.coordinates = coordinates
        self.quartiers = quartiers
        
    def calculate_spatial_errors(self):
        """Calculate prediction errors and their spatial distribution"""
        print("Calcul des erreurs spatiales...")
        
        # Création du DataFrame des erreurs
        self.errors_df = pd.DataFrame({
            'actual': self.actual_values,
            'predicted': self.predictions,
            'quartier': self.quartiers,
            'longitude': self.coordinates['x'],
            'latitude': self.coordinates['y']
        })
        
        # Calcul des erreurs relatives en pourcentage
        self.errors_df['error_pct'] = (
            (self.errors_df['predicted'] - self.errors_df['actual']) / self.errors_df['actual'] * 100
        )
        
        # Calcul des statistiques par quartier
        self.quartier_stats = self.errors_df.groupby('quartier').agg({
            'error_pct': ['mean', 'std', 'count'],
            'actual': 'mean',
            'predicted': 'mean'
        }).round(2)
        
        return self.quartier_stats
    
    def plot_errors_by_quartier(self, viz_path: str, model_name: str) -> None:
        """
        Plot error distribution by quartier and save to appropriate directory
        
        Args:
            viz_path: Base path for visualizations
            model_name: Name of the model (for file naming)
        """
        # Création du dossier spécifique au modèle si nécessaire
        model_viz_path = os.path.join(viz_path, model_name)
        os.makedirs(model_viz_path, exist_ok=True)
        
        # Boxplot des erreurs par quartier
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=self.errors_df, x='quartier', y='error_pct')
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution des erreurs de prédiction par quartier')
        plt.xlabel('Quartier')
        plt.ylabel('Erreur relative (%)')
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_viz_path, 'errors_boxplot_by_district.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
        # Barplot des erreurs moyennes par quartier
        plt.figure(figsize=(15, 8))
        mean_errors = self.errors_df.groupby('quartier')['error_pct'].mean().sort_values()
        
        sns.barplot(x=mean_errors.index, y=mean_errors.values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Erreur moyenne de prédiction par quartier')
        plt.xlabel('Quartier')
        plt.ylabel('Erreur moyenne (%)')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_viz_path, 'errors_barplot_by_district.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
    def plot_price_comparison(self, viz_path: str, model_name: str) -> None:
        """
        Plot actual vs predicted prices by quartier and save to appropriate directory
        
        Args:
            viz_path: Base path for visualizations
            model_name: Name of the model (for file naming)
        """
        # Création du dossier spécifique au modèle si nécessaire
        model_viz_path = os.path.join(viz_path, model_name)
        os.makedirs(model_viz_path, exist_ok=True)
        
        plt.figure(figsize=(15, 8))
        
        # Calcul des prix moyens par quartier
        price_comparison = self.errors_df.groupby('quartier').agg({
            'actual': 'mean',
            'predicted': 'mean'
        }).reset_index()
        
        # Tri par prix réel
        price_comparison = price_comparison.sort_values('actual', ascending=True)
        
        # Création du graphique
        x = range(len(price_comparison))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], price_comparison['actual'], 
                width, label='Prix réel', color='blue', alpha=0.6)
        plt.bar([i + width/2 for i in x], price_comparison['predicted'], 
                width, label='Prix prédit', color='red', alpha=0.6)
        
        plt.xticks(x, price_comparison['quartier'], rotation=45, ha='right')
        plt.title('Comparaison des prix réels et prédits par quartier')
        plt.xlabel('Quartier')
        plt.ylabel('Prix (€)')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(model_viz_path, 'price_comparison_by_district.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()