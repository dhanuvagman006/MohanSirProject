"""
water_storage.py
================
Calculates harvestable water from predicted rainfall for Dakshina Kannada.

Formulas:
    water_collected = rainfall_mm × catchment_m² × runoff_coef × reservoir_eff × 0.001 × 1000
    evaporation_loss = evap_rate_mm_day × catchment_m² × 0.001 × 1000
    net_storage = collected - evaporation
    cumulative = rolling sum of net_storage

Usage:
    from utils.water_storage import calculate_water_storage
    results = calculate_water_storage(predictions, catchment_area_m2=10000)
"""

from typing import List, Dict, Union
from datetime import datetime


def calculate_water_storage(
    predictions: List[Dict[str, Union[str, float]]],
    catchment_area_m2: float = 10000,
    runoff_coefficient: float = 0.75,
    reservoir_efficiency: float = 0.85,
    evaporation_rates: Dict[str, float] = None
) -> List[Dict]:
    """
    Calculate day-wise water storage from rainfall predictions.
    
    Args:
        predictions: List of dicts with 'date' and 'predicted_rainfall_mm'
        catchment_area_m2: Collection area in square meters
        runoff_coefficient: Fraction of rainfall that becomes runoff (0.1-1.0)
        reservoir_efficiency: Storage system efficiency (0.1-1.0)
        evaporation_rates: Dict mapping season to daily evaporation mm
    
    Returns:
        List of dicts with daily storage calculations
    """
    # Default evaporation rates (mm/day) for Dakshina Kannada
    if evaporation_rates is None:
        evaporation_rates = {
            'summer': 2.5,   # Mar-May
            'monsoon': 1.0,  # Jun-Sep
            'post_monsoon': 1.5,  # Oct-Nov
            'winter': 1.2    # Dec-Feb
        }
    
    results = []
    cumulative = 0.0
    
    for pred in predictions:
        date_str = pred['date']
        rainfall_mm = pred['predicted_rainfall_mm']
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        month = date_obj.month
        
        # Determine evaporation rate by season
        if month in [3, 4, 5]:
            evap_rate = evaporation_rates['summer']
        elif month in [6, 7, 8, 9]:
            evap_rate = evaporation_rates['monsoon']
        elif month in [10, 11]:
            evap_rate = evaporation_rates['post_monsoon']
        else:  # 12, 1, 2
            evap_rate = evaporation_rates['winter']
        
        # Calculate water collected (liters)
        # rainfall_mm × m² × 0.001 = m³, × 1000 = liters
        water_collected_L = (
            rainfall_mm * catchment_area_m2 * 
            runoff_coefficient * reservoir_efficiency
        )
        
        # Calculate evaporation loss (liters)
        evaporation_loss_L = evap_rate * catchment_area_m2
        
        # Net storage for this day
        net_storage_L = water_collected_L - evaporation_loss_L
        
        # Update cumulative (can't go negative)
        cumulative = max(0, cumulative + net_storage_L)
        
        results.append({
            'date': date_str,
            'predicted_rainfall_mm': round(rainfall_mm, 2),
            'water_collected_L': round(water_collected_L, 0),
            'evaporation_loss_L': round(evaporation_loss_L, 0),
            'net_storage_L': round(net_storage_L, 0),
            'cumulative_storage_L': round(cumulative, 0),
            'evaporation_rate_mm': evap_rate
        })
    
    return results