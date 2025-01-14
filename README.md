# Dijon Real Estate Price Prediction

A machine learning project to predict real estate prices in Dijon, France, using various property characteristics and spatial features.

## Project Overview

This project implements a comprehensive real estate price prediction system for Dijon, combining traditional property features with spatial analysis and proximity to amenities. The system uses multiple machine learning models including Linear Regression and XGBoost to provide accurate price estimates.

## Features

- **Multiple Model Support**: Implementation of both Linear Regression and XGBoost models
- **Spatial Analysis**: Integration of geographical data including:
  - Distance to public transportation (bus and tram stops)
  - Distance to schools
  - Number of nearby shops (within 150m radius)
  - Neighborhood (quartier) analysis
- **Comprehensive Data Preprocessing**:
  - Handling of missing values
  - Feature encoding
  - Spatial data transformation
  - Outlier removal
- **Model Performance Visualization**:
  - Prediction vs actual value comparisons
  - Feature importance analysis
  - Spatial error distribution
  - Price comparisons by neighborhood

## Project Structure

```
dijon_real_estate/
├── src/
│   ├── real_estate_modeling.py     # Main modeling implementation
│   ├── real_estate_preprocessor.py # Data preprocessing pipeline
│   └── spatial_analysis.py         # Spatial analysis utilities
├── data/
│   ├── raw/                        # Raw GeoJSON data files
│   │   ├── immo.geojson
│   │   ├── bus.geojson
│   │   ├── tram.geojson
│   │   ├── ecoles.geojson
│   │   └── commerces.geojson
│   └── processed/                  # Preprocessed data files
│       ├── features.csv
│       ├── target.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── results/
│   ├── metrics/                    # Model performance metrics
│   │   ├── linear_metrics.json
│   │   └── xgboost_metrics.json
│   ├── preprocessing_stats/        # Data preprocessing statistics
│   │   └── preprocessing_stats.json
│   ├── spatial/                    # Spatial analysis results
│   │   ├── linear_spatial_stats.csv
│   │   └── xgboost_spatial_stats.csv
│   └── visualizations/
│       ├── model_comparison/       # Cross-model visualizations
│       │   ├── predictions_scatter.png
│       │   └── feature_importance.png
│       ├── linear_regression/      # Linear regression specific analysis
│       │   ├── errors_boxplot_by_district.png
│       │   ├── errors_barplot_by_district.png
│       │   └── price_comparison_by_district.png
│       └── xgboost/               # XGBoost specific analysis
│           ├── errors_boxplot_by_district.png
│           ├── errors_barplot_by_district.png
│           └── price_comparison_by_district.png
└── README.md
```

## Prerequisites

The project requires the following Python packages:

- numpy
- pandas
- geopandas
- scikit-learn
- xgboost
- matplotlib
- seaborn

## Data Requirements

All input data is sourced from the [Dijon Open Data Portal](https://dijon-metropole.opendatasoft.com/pages/portal-explore/).

The system expects GeoJSON files in the `data/raw/` directory for:
- Real estate transactions (`immo.geojson`)
- Bus stops (`bus.geojson`)
- Tram stops (`tram.geojson`)
- Schools (`ecoles.geojson`)
- Shops/businesses (`commerces.geojson`)

## Usage

1. Prepare your data files in the `data/raw/` directory
2. Run the preprocessing pipeline:
```python
from src.real_estate_preprocessor import RealEstatePreprocessor

preprocessor = RealEstatePreprocessor('path/to/data')
X_train, X_test, y_train, y_test, quartiers_train, quartiers_test = preprocessor.main()
```

3. Train and evaluate models:
```python
from src.real_estate_modeling import RealEstateModeling

modeling = RealEstateModeling(X_train, X_test, y_train, y_test)
modeling.train_linear_regression()
modeling.train_xgboost()
modeling.calculate_metrics()
```

4. Visualize results:
```python
modeling.plot_predictions()
modeling.plot_feature_importance()
modeling.analyze_spatial_predictions(quartiers_test)
```

## Model Features

The system uses various features for prediction, including:
- Property characteristics (type, size, number of rooms)
- Location data (coordinates, commune)
- Proximity to amenities (schools, public transport, shops)
- Temporal features (month and year of transaction)

## Data Processing and Model Tuning

### Data Filtering
To ensure robust predictions for standard real estate properties, the system implements several filtering steps:
- Removal of large plots (>10,000 m²)
- Filtering of extreme property prices (outside 1st-99th percentiles)
- Filtering of extreme price per square meter values (outside 1st-99th percentiles)

This filtering process removes approximately 3.2% of the initial dataset, focusing the analysis on representative properties and excluding atypical assets (luxury hotels, exceptional properties, etc.).

## Performance Metrics & Interpretation

The system evaluates models using:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Spatial error distribution by neighborhood

### Model Performance Analysis

After data filtering, both models show excellent predictive capabilities, with XGBoost significantly outperforming Linear Regression:

- **Linear Regression**:
  - Test R²: 0.743 (explaining 74.3% of price variance)
  - Test RMSE: 46,591€
  - Test MAE: 32,854€
  - Shows excellent consistency between training and test performance (R² diff: 0.002)

- **XGBoost** (with optimized hyperparameters):
  - Test R²: 0.819 (explaining 81.9% of price variance)
  - Test RMSE: 39,117€ 
  - Test MAE: 25,974€
  - Shows strong performance with reasonable generalization gap

### Spatial Analysis

The neighborhood analysis shows remarkable improvement after data filtering:

1. **Price Prediction Accuracy by District**:
   - XGBoost shows outstanding consistency across all districts (error ranges 0.8-10.3%)
   - Linear Regression performs well in most areas but shows some weakness in specific districts
   - Both models excel in districts like Varennes-Toison d'Or (0-0.8% error) and Université (3.3-3.5% error)

2. **District-Specific Performance**:
   - Most districts show stable predictions with error standard deviations around 25-35%
   - Fontaine d'Ouche shows notably different results between models (Linear: 38.8% vs XGBoost: 8% error)
   - Montchapet, with high transaction volume (1,153 properties), shows very stable predictions

These results demonstrate that the models, particularly XGBoost, provide reliable predictions across all neighborhoods after proper data filtering. The Linear Regression model still shows some limitations in specific areas, suggesting that local market dynamics might not be entirely linear.

## Spatial Analysis

The spatial analysis component provides:
- Error distribution visualization by neighborhood
- Comparison of actual vs predicted prices by area
- Detailed statistics for each neighborhood
- Identification of areas with systematic under/over-prediction

## Contributing

Feel free to submit issues and enhancement requests!