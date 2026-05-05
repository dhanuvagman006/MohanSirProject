"""
irrigation.py
=============
Generates smart irrigation schedules based on rainfall predictions and crop needs.

Logic:
    effective_rainfall = min(predicted_rainfall, crop_water_requirement)
    irrigation_needed = max(0, crop_water_requirement - effective_rainfall)
    irrigation_volume = irrigation_needed_mm × field_area_m² × 0.001 × 1000
    action = "irrigate" if irrigation_needed > 0.5mm else "skip"

Usage:
    from utils.irrigation import calculate_irrigation_schedule
    schedule = calculate_irrigation_schedule(predictions, crop='paddy', field_area_m2=5000)
"""

from typing import List, Dict, Union
from datetime import datetime


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


def calculate_irrigation_schedule(
    predictions: List[Dict[str, Union[str, float]]],
    crop: str,
    field_area_m2: float,
    crop_water_req_mm: float = None
) -> List[Dict]:
    """
    Generate day-wise irrigation recommendations.
    
    Args:
        predictions: List of dicts with 'date' and 'predicted_rainfall_mm'
        crop: Crop type (must be in CROP_WATER_REQUIREMENTS)
        field_area_m2: Field area in square meters
        crop_water_req_mm: Optional custom water requirement (overrides default)
    
    Returns:
        List of dicts with daily irrigation schedule
    """
    # Get crop water requirement
    if crop_water_req_mm is None:
        crop_info = CROP_WATER_REQUIREMENTS.get(crop.lower(), {})
        crop_water_req_mm = crop_info.get('default', 5)  # Default 5mm/day
    
    results = []
    total_water_saved = 0.0
    
    for pred in predictions:
        date_str = pred['date']
        rainfall_mm = pred['predicted_rainfall_mm']
        
        # Calculate effective rainfall (capped at crop need)
        effective_rainfall = min(rainfall_mm, crop_water_req_mm)
        
        # Calculate irrigation needed
        irrigation_needed_mm = max(0, crop_water_req_mm - effective_rainfall)
        
        # Convert to liters: mm × m² × 0.001 × 1000 = liters
        irrigation_volume_L = irrigation_needed_mm * field_area_m2
        
        # Determine action (threshold: 0.5mm to avoid micro-irrigation)
        schedule_action = "irrigate" if irrigation_needed_mm > 0.5 else "skip"
        
        # Calculate water saved by using rainfall instead of irrigation
        water_savings_L = effective_rainfall * field_area_m2
        total_water_saved += water_savings_L
        
        results.append({
            'date': date_str,
            'crop': crop,
            'rainfall_mm': round(rainfall_mm, 2),
            'effective_rainfall_mm': round(effective_rainfall, 2),
            'irrigation_needed_mm': round(irrigation_needed_mm, 2),
            'irrigation_volume_L': round(irrigation_volume_L, 0),
            'schedule_action': schedule_action,
            'water_savings_L': round(water_savings_L, 0)
        })
    
    return results


def get_crop_info(crop: str) -> Dict:
    """Get detailed information about a crop's water requirements."""
    return CROP_WATER_REQUIREMENTS.get(crop.lower(), {
        'min': 4, 'max': 6, 'default': 5,
        'note': 'Using default values - specify crop from supported list'
    })


def get_supported_crops() -> List[Dict]:
    """Return list of supported crops with their water requirements."""
    return [
        {
            'key': key,
            'name': key.capitalize(),
            **info
        }
        for key, info in CROP_WATER_REQUIREMENTS.items()
    ]