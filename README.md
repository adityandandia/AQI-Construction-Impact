# AQI Construction Impact Predictor

## Research Question
Does construction intensity contribute to short-term AQI variability 
beyond what weather factors alone can explain?

## Dataset
- 21 Indian cities
- 2015–2020 (6 years)
- 24,699 data points
- Sources: CPCB AQI data + Open-Meteo weather API

## Methodology
Two-model comparison using LightGBM:
- **Model A (Baseline):** Weather-only features
- **Model B (Extended):** Weather + construction intensity

## Results
| Metric | Model A | Model B | Improvement |
|--------|---------|---------|-------------|
| RMSE   | 31.94   | 29.53   | ↓ 7.5%     |
| MAE    | 20.78   | 19.40   | ↓ 6.6%     |
| R²     | 0.353   | 0.447   | ↑ 26.6%    |

Spike recall improved from 0.66 → 0.84 (+27%).

**H2 Supported:** Construction intensity adds predictive signal beyond weather.

## Stack
Python · LightGBM · FastAPI · Open-Meteo API · Pandas · Scikit-learn

## Limitations
- Construction intensity is simulated, not from real permit data
- City-level aggregation only
- Observational model — correlation, not causation

## Next Steps
- Faculty collaboration for real construction permit data access
- GIS spatial modeling
- Real-time pipeline: FastAPI + Docker + AWS
