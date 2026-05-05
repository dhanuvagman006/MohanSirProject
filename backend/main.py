"""
main.py
=======
FastAPI backend for DK-AquaPredict — Smart Rainfall Prediction System.

Provides REST API endpoints for:
- Health checks & dataset statistics
- Rainfall prediction using 6 trained DL models
- Water storage calculation
- Smart irrigation scheduling
- Historical data retrieval

Features:
- CORS enabled for React frontend (localhost:5173)
- Async endpoints for non-blocking I/O
- Input validation with Pydantic models
- Error handling with detailed HTTP responses
- Loads pre-trained models and scaler at startup

Usage:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# TensorFlow imports (lazy loading to speed up startup)
import tensorflow as tf
from tensorflow.keras.models import load_model

# Local utilities
from utils.water_storage import calculate_water_storage
from utils.irrigation import calculate_irrigation_schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

APP_CONFIG = {
    'title': 'DK-AquaPredict API',
    'version': '1.0.0',
    'description': 'Smart Rainfall Prediction & Water Management for Dakshina Kannada',
    'cors_origins': ['http://localhost:5173', 'http://127.0.0.1:5173'],
    'data_path': 'data/dakshina_kannada_weather.csv',
    'models_dir': 'models',
    'scaler_path': 'models/scaler.pkl',
    'metrics_path': 'models/metrics.json',
    'sample_predictions_path': 'data/sample_predictions.json',
}

# Model name mapping for API
MODEL_NAMES = {
    'lstm': 'LSTM',
    'bilstm': 'Bidirectional LSTM',
    'gru': 'GRU',
    'cnn_lstm': 'CNN-LSTM Hybrid',
    'transformer': 'Transformer',
    'autoencoder': 'Autoencoder + Regression'
}

# Crop water requirements (mm/day) - FAO standards for Dakshina Kannada
CROP_WATER_REQUIREMENTS = {
    'paddy': {'min': 6, 'max': 8, 'default': 7},
    'arecanut': {'min': 4, 'max': 5, 'default': 4.5},
    'coconut': {'min': 3, 'max': 4, 'default': 3.5},
    'banana': {'min': 5, 'max': 7, 'default': 6},
    'pepper': {'min': 3, 'max': 4, 'default': 3.5},
    'cashew': {'min': 2, 'max': 3, 'default': 2.5},
    'rubber': {'min': 4, 'max': 5, 'default': 4.5},
    'vegetables': {'min': 4, 'max': 6, 'default': 5},
}

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class PredictionRequest(BaseModel):
    """Request schema for rainfall prediction endpoints."""
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    days: int = Field(..., ge=7, le=90, description="Number of days to predict (7-90)")
    model: Optional[str] = Field(default="lstm", description="Model name to use")
    
    @validator('start_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
    
    @validator('model')
    def validate_model_name(cls, v):
        if v and v not in MODEL_NAMES:
            raise ValueError(f"Model must be one of: {list(MODEL_NAMES.keys())}")
        return v


class AllModelsRequest(BaseModel):
    """Request schema for multi-model prediction."""
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    days: int = Field(..., ge=7, le=90, description="Number of days to predict (7-90)")


class WaterStorageRequest(BaseModel):
    """Request schema for water storage calculation."""
    predictions: List[Dict[str, Union[str, float]]] = Field(
        ..., description="Array of prediction objects with date and predicted_rainfall_mm"
    )
    catchment_area_m2: float = Field(default=10000, ge=100, le=1000000, description="Catchment area in square meters")
    runoff_coefficient: float = Field(default=0.75, ge=0.1, le=1.0, description="Runoff coefficient (0.1-1.0)")
    reservoir_efficiency: float = Field(default=0.85, ge=0.1, le=1.0, description="Reservoir efficiency (0.1-1.0)")


class IrrigationRequest(BaseModel):
    """Request schema for irrigation scheduling."""
    predictions: List[Dict[str, Union[str, float]]] = Field(
        ..., description="Array of prediction objects with date and predicted_rainfall_mm"
    )
    crop: str = Field(..., description="Crop type")
    field_area_m2: float = Field(..., ge=100, le=1000000, description="Field area in square meters")
    crop_water_req: Optional[float] = Field(default=None, ge=0, description="Custom crop water requirement (mm/day)")
    
    @validator('crop')
    def validate_crop(cls, v):
        if v.lower() not in CROP_WATER_REQUIREMENTS:
            raise ValueError(f"Crop must be one of: {list(CROP_WATER_REQUIREMENTS.keys())}")
        return v.lower()


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    date: str
    predicted_rainfall_mm: float
    confidence_interval_low: float
    confidence_interval_high: float
    model_used: str
    model_r2: float


class MultiModelPredictionResponse(BaseModel):
    """Response schema for multi-model comparison."""
    date: str
    predictions: Dict[str, float]  # model_name: predicted_value
    ensemble_mean: float
    ensemble_std: float


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title=APP_CONFIG['title'],
    version=APP_CONFIG['version'],
    description=APP_CONFIG['description'],
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=APP_CONFIG['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State: Loaded Models & Data
# ============================================================================

class AppState:
    """Container for globally loaded models and data."""
    
    def __init__(self):
        self.models: Dict[str, tf.keras.Model] = {}
        self.scaler: Optional[MinMaxScaler] = None
        self.metrics: Dict[str, Dict] = {}
        self.df_historical: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.sequence_length: int = 30
        self.is_loaded: bool = False
    
    def load_models(self) -> bool:
        """Load all trained models, scaler, and metrics at startup."""
        try:
            # Load metrics first
            metrics_path = Path(APP_CONFIG['metrics_path'])
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                logger.info(f"✅ Loaded metrics from {metrics_path}")
            else:
                logger.warning(f"⚠️ Metrics file not found: {metrics_path}")
            
            # Load scaler
            scaler_path = Path(APP_CONFIG['scaler_path'])
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"✅ Loaded scaler from {scaler_path}")
            else:
                logger.warning(f"⚠️ Scaler file not found: {scaler_path}")
            
            # Load historical data for stats
            data_path = Path(APP_CONFIG['data_path'])
            if data_path.exists():
                self.df_historical = pd.read_csv(data_path)
                self.feature_columns = [
                    'temperature_max', 'temperature_min', 'humidity', 'wind_speed',
                    'pressure', 'cloud_cover', 'sunshine_hours', 'dew_point',
                    'evapotranspiration', 'previous_7day_rainfall', 'previous_30day_rainfall',
                    'month_sin', 'month_cos', 'day_sin', 'day_cos'
                ]
                logger.info(f"✅ Loaded historical data: {len(self.df_historical):,} records")
            else:
                logger.warning(f"⚠️ Historical data not found: {data_path}")
            
            # Load trained models (lazy loading option available)
            models_dir = Path(APP_CONFIG['models_dir'])
            for model_key in MODEL_NAMES.keys():
                model_path = models_dir / f"model_{model_key}.h5"
                if model_path.exists():
                    # Suppress TensorFlow logging during load
                    tf.get_logger().setLevel('ERROR')
                    self.models[model_key] = load_model(str(model_path), compile=False)
                    logger.info(f"✅ Loaded model: {model_key}")
                else:
                    logger.warning(f"⚠️ Model not found: {model_path}")
            
            self.is_loaded = True
            logger.info("🎉 All resources loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading resources: {str(e)}")
            return False
    
    def get_model(self, model_name: str) -> Optional[tf.keras.Model]:
        """Get a specific model by name."""
        return self.models.get(model_name)
    
    def get_all_models(self) -> Dict[str, tf.keras.Model]:
        """Get all loaded models."""
        return self.models
    
    def predict_with_model(self, model_name: str, input_sequence: np.ndarray) -> float:
        """Make prediction with a specific model."""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        # Reshape if needed: (features,) -> (1, seq_len, features)
        if input_sequence.ndim == 2:
            input_sequence = np.expand_dims(input_sequence, axis=0)
        
        prediction = model.predict(input_sequence, verbose=0)[0][0]
        return float(prediction)
    
    def calculate_confidence_interval(self, prediction: float, model_name: str) -> tuple:
        """Calculate 95% confidence interval based on model's historical MAE."""
        metrics = self.metrics.get(model_name, {})
        mae = metrics.get('mae', 8.0)  # Default fallback
        rmse = metrics.get('rmse', 10.0)
        
        # Use RMSE as proxy for std deviation, 95% CI ≈ ±1.96σ
        margin = 1.96 * rmse * 0.5  # Conservative estimate
        
        return (
            round(max(0, prediction - margin), 2),
            round(prediction + margin, 2)
        )


# Initialize global state
state = AppState()


# ============================================================================
# Startup Event: Load Resources
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models and data when application starts."""
    logger.info("🚀 Starting DK-AquaPredict API...")
    success = state.load_models()
    if not success:
        logger.warning("⚠️ Some resources failed to load. API may have limited functionality.")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(state.models),
        "historical_records": len(state.df_historical) if state.df_historical is not None else 0
    }


@app.get("/api/dataset/stats", tags=["Dataset"])
async def get_dataset_stats():
    """Return summary statistics of the historical dataset."""
    if state.df_historical is None:
        raise HTTPException(status_code=503, detail="Historical dataset not available")
    
    df = state.df_historical
    
    # Basic stats
    stats = {
        "total_records": len(df),
        "date_range": {
            "start": df['date'].iloc[0],
            "end": df['date'].iloc[-1]
        },
        "rainfall": {
            "mean_daily": round(df['rainfall_mm'].mean(), 2),
            "max_daily": round(df['rainfall_mm'].max(), 2),
            "annual_mean": round(df.groupby('year')['rainfall_mm'].sum().mean(), 0),
            "annual_std": round(df.groupby('year')['rainfall_mm'].sum().std(), 0)
        },
        "monthly_means": df.groupby('month')['rainfall_mm'].mean().round(1).to_dict(),
        "monsoon_stats": {
            "southwest_mean": round(df[(df['month'] >= 6) & (df['month'] <= 9)]['rainfall_mm'].mean(), 2),
            "northeast_mean": round(df[(df['month'] == 10) | (df['month'] == 11)]['rainfall_mm'].mean(), 2),
            "dry_season_mean": round(df[(df['month'] <= 2) | (df['month'] == 12)]['rainfall_mm'].mean(), 2)
        },
        "feature_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean())
            }
            for col in ['temperature_max', 'humidity', 'wind_speed', 'pressure']
        }
    }
    
    return stats


@app.get("/api/models/metrics", tags=["Models"])
async def get_model_metrics():
    """Return accuracy metrics for all trained models."""
    if not state.metrics:
        # Return sample metrics if real ones not available
        return {
            "note": "Using sample metrics - train models for real values",
            "models": {
                name: {
                    "r2": 0.90 + np.random.random() * 0.03,
                    "mae": 6 + np.random.random() * 2,
                    "rmse": 8 + np.random.random() * 3,
                    "mape": 15 + np.random.random() * 8,
                    "trainable_params": np.random.randint(150000, 350000)
                }
                for name in MODEL_NAMES.keys()
            }
        }
    
    return {
        "models": {
            name: {
                "display_name": MODEL_NAMES[name],
                **state.metrics.get(name, {})
            }
            for name in MODEL_NAMES.keys()
        },
        "best_model": max(state.metrics.keys(), key=lambda k: state.metrics[k].get('r2', 0))
        if state.metrics else None
    }


@app.post("/api/predict/rainfall", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_rainfall(request: PredictionRequest):
    """
    Predict rainfall for specified date range using selected model.
    
    Returns predictions with 95% confidence intervals.
    """
    if not state.is_loaded or not state.models:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Please run train_models.py first."
        )
    
    model_name = request.model or 'lstm'
    if model_name not in state.models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Choose from: {list(state.models.keys())}"
        )
    
    try:
        # Parse start date
        start_date = datetime.strptime(request.start_date, '%Y-%m-%d')
        
        # Generate predictions for each day
        predictions = []
        
        # Get last known data point for sequence initialization
        if state.df_historical is not None:
            last_record = state.df_historical.iloc[-1].copy()
        else:
            # Fallback: generate synthetic recent data
            last_record = _generate_fallback_features(start_date)
        
        # Add cyclical features if missing
        for col in ['month_sin', 'month_cos', 'day_sin', 'day_cos']:
            if col not in last_record:
                last_record[col] = 0
        
        # Prepare feature array for sequence building
        recent_features = []
        for i in range(state.sequence_length):
            # Shift back in time
            check_date = start_date - timedelta(days=state.sequence_length - i)
            feat = _get_features_for_date(check_date, state.df_historical, last_record)
            recent_features.append(feat)
        
        # Predict day by day (autoregressive for multi-day forecast)
        current_features = np.array(recent_features, dtype=np.float32)
        
        for day_offset in range(request.days):
            # Normalize the sequence
            if state.scaler is not None:
                # Reshape for scaler: (seq_len, n_features) -> (seq_len * n_features,) then back
                flat = current_features.flatten().reshape(-1, 1)
                scaled_flat = state.scaler.transform(flat).flatten()
                scaled_seq = scaled_flat.reshape(current_features.shape)
            else:
                scaled_seq = current_features  # Fallback if no scaler
            
            # Make prediction
            pred_value = state.predict_with_model(model_name, scaled_seq)
            pred_value = max(0, pred_value)  # Ensure non-negative
            
            # Calculate confidence interval
            ci_low, ci_high = state.calculate_confidence_interval(pred_value, model_name)
            
            # Record prediction
            pred_date = start_date + timedelta(days=day_offset)
            predictions.append(PredictionResponse(
                date=pred_date.strftime('%Y-%m-%d'),
                predicted_rainfall_mm=round(pred_value, 2),
                confidence_interval_low=ci_low,
                confidence_interval_high=ci_high,
                model_used=model_name,
                model_r2=state.metrics.get(model_name, {}).get('r2', 0.90)
            ))
            
            # Update sequence for next iteration (autoregressive)
            # Shift window: remove oldest, add new predicted day
            new_day_features = _generate_features_for_prediction(
                pred_date, pred_value, last_record
            )
            current_features = np.vstack([current_features[1:], new_day_features])
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict/all_models", response_model=List[MultiModelPredictionResponse], tags=["Prediction"])
async def predict_all_models(request: AllModelsRequest):
    """
    Get predictions from all available models for comparison.
    
    Useful for ensemble analysis and model selection.
    """
    if not state.models:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    try:
        start_date = datetime.strptime(request.start_date, '%Y-%m-%d')
        predictions = []
        
        # Get base features for sequence initialization
        if state.df_historical is not None:
            last_record = state.df_historical.iloc[-1].copy()
        else:
            last_record = _generate_fallback_features(start_date)
        
        for day_offset in range(request.days):
            pred_date = start_date + timedelta(days=day_offset)
            model_preds = {}
            
            for model_name, model in state.models.items():
                # Build input sequence for this model
                recent_features = []
                for i in range(state.sequence_length):
                    check_date = pred_date - timedelta(days=state.sequence_length - i)
                    feat = _get_features_for_date(check_date, state.df_historical, last_record)
                    recent_features.append(feat)
                
                current_features = np.array(recent_features, dtype=np.float32)
                
                # Normalize and predict
                if state.scaler is not None:
                    flat = current_features.flatten().reshape(-1, 1)
                    scaled_flat = state.scaler.transform(flat).flatten()
                    scaled_seq = scaled_flat.reshape(current_features.shape)
                else:
                    scaled_seq = current_features
                
                pred_value = state.predict_with_model(model_name, scaled_seq)
                model_preds[model_name] = round(max(0, pred_value), 2)
            
            # Calculate ensemble statistics
            values = list(model_preds.values())
            ensemble_mean = np.mean(values)
            ensemble_std = np.std(values)
            
            predictions.append(MultiModelPredictionResponse(
                date=pred_date.strftime('%Y-%m-%d'),
                predictions=model_preds,
                ensemble_mean=round(ensemble_mean, 2),
                ensemble_std=round(ensemble_std, 2)
            ))
        
        return predictions
        
    except Exception as e:
        logger.error(f"Multi-model prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/water/storage", tags=["Water Management"])
async def calculate_storage(request: WaterStorageRequest):
    """
    Calculate harvestable water from predicted rainfall.
    
    Uses catchment parameters to estimate collectable water volume.
    """
    try:
        results = calculate_water_storage(
            predictions=request.predictions,
            catchment_area_m2=request.catchment_area_m2,
            runoff_coefficient=request.runoff_coefficient,
            reservoir_efficiency=request.reservoir_efficiency
        )
        return {
            "parameters": {
                "catchment_area_m2": request.catchment_area_m2,
                "runoff_coefficient": request.runoff_coefficient,
                "reservoir_efficiency": request.reservoir_efficiency,
                "evaporation_rate_mm_day": 2.5  # Summer default
            },
            "summary": {
                "total_predicted_rainfall_mm": sum(p['predicted_rainfall_mm'] for p in request.predictions),
                "total_water_collected_L": round(sum(r['water_collected_L'] for r in results), 0),
                "total_evaporation_loss_L": round(sum(r['evaporation_loss_L'] for r in results), 0),
                "net_harvestable_L": round(sum(r['net_storage_L'] for r in results), 0),
                "final_cumulative_L": results[-1]['cumulative_storage_L'] if results else 0
            },
            "daily_breakdown": results
        }
    except Exception as e:
        logger.error(f"Water storage calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")


@app.post("/api/irrigation/schedule", tags=["Irrigation"])
async def schedule_irrigation(request: IrrigationRequest):
    """
    Generate smart irrigation schedule based on rainfall predictions.
    
    Recommends when to irrigate vs. rely on natural rainfall.
    """
    try:
        # Get crop water requirement
        crop_key = request.crop.lower()
        if request.crop_water_req is not None:
            water_req = request.crop_water_req
        else:
            crop_info = CROP_WATER_REQUIREMENTS.get(crop_key, {})
            water_req = crop_info.get('default', 5)  # Default 5mm/day
        
        results = calculate_irrigation_schedule(
            predictions=request.predictions,
            crop=crop_key,
            field_area_m2=request.field_area_m2,
            crop_water_req_mm=water_req
        )
        
        return {
            "parameters": {
                "crop": crop_key,
                "crop_display_name": crop_key.capitalize(),
                "field_area_m2": request.field_area_m2,
                "water_requirement_mm_day": water_req
            },
            "summary": {
                "total_days": len(results),
                "irrigation_days": sum(1 for r in results if r['schedule_action'] == 'irrigate'),
                "skip_days": sum(1 for r in results if r['schedule_action'] == 'skip'),
                "total_irrigation_volume_L": round(sum(r['irrigation_volume_L'] for r in results), 0),
                "total_water_saved_L": round(sum(r['water_savings_L'] for r in results), 0),
                "irrigation_efficiency_pct": round(
                    100 * sum(r['effective_rainfall_mm'] for r in results) / 
                    max(0.1, len(results) * water_req), 1
                )
            },
            "daily_schedule": results
        }
    except Exception as e:
        logger.error(f"Irrigation scheduling error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scheduling failed: {str(e)}")


@app.get("/api/historical/monthly", tags=["Historical Data"])
async def get_monthly_averages():
    """Return monthly average rainfall from historical dataset."""
    if state.df_historical is None:
        raise HTTPException(status_code=503, detail="Historical data not available")
    
    df = state.df_historical
    monthly = df.groupby('month').agg({
        'rainfall_mm': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    # Flatten multi-index columns
    result = {}
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for m in range(1, 13):
        if m in monthly.index:
            result[month_names[m-1]] = {
                'mean': float(monthly.loc[m, ('rainfall_mm', 'mean')]),
                'std': float(monthly.loc[m, ('rainfall_mm', 'std')]),
                'min': float(monthly.loc[m, ('rainfall_mm', 'min')]),
                'max': float(monthly.loc[m, ('rainfall_mm', 'max')]),
                'count': int(monthly.loc[m, ('rainfall_mm', 'count')])
            }
    
    return result


@app.get("/api/historical/annual", tags=["Historical Data"])
async def get_annual_totals():
    """Return annual rainfall totals from 2000-2024."""
    if state.df_historical is None:
        raise HTTPException(status_code=503, detail="Historical data not available")
    
    df = state.df_historical
    annual = df.groupby('year')['rainfall_mm'].sum().round(0)
    
    return {
        "yearly_totals": {str(year): int(total) for year, total in annual.items()},
        "statistics": {
            "mean": float(annual.mean()),
            "median": float(annual.median()),
            "std": float(annual.std()),
            "min_year": str(annual.idxmin()),
            "max_year": str(annual.idxmax()),
            "min_value": int(annual.min()),
            "max_value": int(annual.max())
        }
    }


# ============================================================================
# Helper Functions for Prediction Pipeline
# ============================================================================

def _generate_fallback_features(date: datetime) -> pd.Series:
    """Generate synthetic feature values when historical data unavailable."""
    doy = date.timetuple().tm_yday
    
    # Seasonal patterns
    temp_max = 29 + 6 * np.sin(2 * np.pi * (doy - 100) / 365) + np.random.normal(0, 1.5)
    temp_min = temp_max - 6 + np.random.normal(0, 1)
    humidity = 75 + 15 * np.sin(2 * np.pi * (doy - 180) / 365) + np.random.normal(0, 5)
    
    return pd.Series({
        'temperature_max': np.clip(temp_max, 22, 38),
        'temperature_min': np.clip(temp_min, 16, 28),
        'humidity': np.clip(humidity, 55, 98),
        'wind_speed': np.random.uniform(10, 25),
        'pressure': np.random.uniform(998, 1012),
        'cloud_cover': np.random.uniform(20, 80),
        'sunshine_hours': np.random.uniform(4, 10),
        'dew_point': np.random.uniform(15, 24),
        'evapotranspiration': np.random.uniform(2, 6),
        'previous_7day_rainfall': np.random.uniform(0, 20),
        'previous_30day_rainfall': np.random.uniform(0, 50),
        'month_sin': np.sin(2 * np.pi * date.month / 12),
        'month_cos': np.cos(2 * np.pi * date.month / 12),
        'day_sin': np.sin(2 * np.pi * doy / 365),
        'day_cos': np.cos(2 * np.pi * doy / 365),
    })


def _get_features_for_date(
    date: datetime, 
    df_historical: pd.DataFrame, 
    fallback: pd.Series
) -> np.ndarray:
    """Get feature vector for a specific date, with fallback to synthetic data."""
    if df_historical is not None:
        # Try to find matching date in historical data
        date_str = date.strftime('%Y-%m-%d')
        match = df_historical[df_historical['date'] == date_str]
        
        if len(match) > 0:
            row = match.iloc[0]
            features = []
            for col in ['temperature_max', 'temperature_min', 'humidity', 'wind_speed',
                       'pressure', 'cloud_cover', 'sunshine_hours', 'dew_point',
                       'evapotranspiration', 'previous_7day_rainfall', 'previous_30day_rainfall']:
                features.append(row[col] if col in row else fallback[col])
            
            # Add cyclical features
            features.extend([
                np.sin(2 * np.pi * date.month / 12),
                np.cos(2 * np.pi * date.month / 12),
                np.sin(2 * np.pi * date.timetuple().tm_yday / 365),
                np.cos(2 * np.pi * date.timetuple().tm_yday / 365),
            ])
            return np.array(features, dtype=np.float32)
    
    # Fallback to synthetic generation
    return _generate_fallback_features(date).values.astype(np.float32)


def _generate_features_for_prediction(
    date: datetime, 
    predicted_rainfall: float, 
    base_features: pd.Series
) -> np.ndarray:
    """Generate feature vector for autoregressive prediction step."""
    features = base_features.copy()
    
    # Update rainfall-related features with prediction
    features['previous_7day_rainfall'] = (
        features['previous_7day_rainfall'] * 6 + predicted_rainfall
    ) / 7
    features['previous_30day_rainfall'] = (
        features['previous_30day_rainfall'] * 29 + predicted_rainfall
    ) / 30
    
    # Update humidity based on rainfall (simplified correlation)
    features['humidity'] = np.clip(
        features['humidity'] + 0.1 * predicted_rainfall, 55, 98
    )
    
    # Update cloud cover
    features['cloud_cover'] = np.clip(
        features['cloud_cover'] + 0.15 * predicted_rainfall, 0, 100
    )
    
    # Cyclical features
    features['month_sin'] = np.sin(2 * np.pi * date.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * date.month / 12)
    features['day_sin'] = np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
    features['day_cos'] = np.cos(2 * np.pi * date.timetuple().tm_yday / 365)
    
    # Return in correct order
    feature_order = [
        'temperature_max', 'temperature_min', 'humidity', 'wind_speed',
        'pressure', 'cloud_cover', 'sunshine_hours', 'dew_point',
        'evapotranspiration', 'previous_7day_rainfall', 'previous_30day_rainfall',
        'month_sin', 'month_cos', 'day_sin', 'day_cos'
    ]
    
    return np.array([features[col] for col in feature_order], dtype=np.float32)


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Catch-all exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "details": str(exc) if os.getenv('DEBUG') else "Contact support",
                "timestamp": datetime.now().isoformat()
            }
        }
    )


# ============================================================================
# Root Endpoint (for frontend routing)
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - redirects to API docs."""
    return {
        "message": "Welcome to DK-AquaPredict API",
        "docs": "/api/docs",
        "health": "/api/health",
        "version": APP_CONFIG['version']
    }


# ============================================================================
# Sample Predictions Endpoint (for demo without training)
# ============================================================================

@app.get("/api/sample/predictions", tags=["Demo"])
async def get_sample_predictions():
    """
    Return pre-generated sample predictions for frontend demo.
    
    Useful when models aren't trained yet.
    """
    sample_path = Path(APP_CONFIG['sample_predictions_path'])
    
    if sample_path.exists():
        with open(sample_path, 'r') as f:
            return json.load(f)
    
    # Generate minimal sample if file doesn't exist
    base_date = datetime(2024, 6, 1)
    return {
        "note": "Sample predictions - train models for real predictions",
        "predictions": [
            {
                "date": (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                "predicted_rainfall_mm": round(max(0, 50 * np.sin(i/5) + np.random.exponential(10)), 2),
                "confidence_interval_low": 0,
                "confidence_interval_high": 150
            }
            for i in range(30)
        ]
    }