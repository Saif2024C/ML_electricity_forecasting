# Electricity Consumption Forecasting using Neural Sequence Models

This project investigates deep learning approaches for multi-horizon residential electricity consumption forecasting using the UCI Individual Household Electric Power Consumption dataset. Two neural sequence architectures, Long Short-Term Memory (LSTM) networks and Transformer encoders, were implemented and evaluated under an identical preprocessing and training pipeline to enable a controlled architectural comparison.

## Objectives
- Forecast residential electricity consumption over a 24-hour horizon
- Compare recurrent (LSTM) and attention-based (Transformer) sequence models
- Analyze convergence behaviour and forecasting accuracy
- Investigate residual error distributions and evaluation metrics

## Workflow

### Data Preprocessing
- Converted raw measurements into a unified datetime index
- Aggregated minute-level observations to hourly resolution
- Handled missing values using forward-fill interpolation
- Standardized features using training-set statistics only to prevent data leakage

### Feature Engineering
Constructed multivariate input features including:
- Global reactive power
- Voltage
- Current intensity
- Sub-metering variables
- Cyclical hour-of-day and day-of-week encodings

A sliding-window supervised learning formulation was used with:
- Lookback window: 168 hours
- Forecast horizon: 24 hours

### Dataset Construction
- Chronological train/validation/test split (70/15/15)
- PyTorch Dataset and DataLoader pipeline
- Windowed multivariate time-series generation

### Model Architectures

#### LSTM Forecaster
- Multi-layer LSTM encoder
- Dense forecasting head
- Gradient clipping for training stability

#### Transformer Forecaster
- Encoder-only Transformer architecture
- Sinusoidal positional encoding
- Multi-head self-attention mechanism
- GELU activations and layer normalization

### Training
- PyTorch implementation
- AdamW optimizer
- SmoothL1 loss
- Validation-based model selection
- RMSE and MAE evaluation on inverse-scaled predictions

### Residual Diagnostics
- Gaussian vs Laplace residual fitting
- Q-Q analysis
- Residual autocorrelation analysis
- Likelihood-based metric justification

## Assumptions
- Electricity consumption exhibits learnable temporal structure
- Temporal dependencies can be captured using fixed-length sliding windows
- Chronological splitting preserves causality and prevents information leakage
- Forecast errors possess finite variance, justifying MAE/RMSE evaluation
- Local stationarity approximately holds within forecasting windows

## Results
- Both models successfully captured temporal consumption patterns and achieved stable convergence
- The Transformer model converged faster and achieved lower forecasting error than the LSTM baseline

| Model | RMSE (kW) |
|---|---|
| LSTM | 1.053 |
| Transformer | 0.983 |

Residual diagnostics suggested approximately Gaussian error behaviour, supporting RMSE as the primary evaluation metric.

## Technologies Used
- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Repository Contents
- Data preprocessing pipeline
- LSTM forecasting model
- Transformer forecasting model
- Training and evaluation scripts
- Residual diagnostic analysis
- Visualization utilities
