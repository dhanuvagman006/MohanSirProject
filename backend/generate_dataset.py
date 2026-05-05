import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional

np.random.seed(42)

REGION = "Dakshina Kannada"
START_YEAR = 2000
END_YEAR = 2024
ANNUAL_RAINFALL_TARGET_MM = (3500, 4500)

TEMP_MAX_RANGE = (22, 38)
TEMP_MIN_RANGE = (16, 28)
HUMIDITY_RANGE = (55, 98)
WIND_SPEED_RANGE = (5, 35)
PRESSURE_RANGE = (995, 1015)
CLOUD_COVER_RANGE = (0, 100)
SUNSHINE_RANGE = (0, 12)
DEW_POINT_RANGE = (10, 26)
EVAPOTRANSPIRATION_RANGE = (1, 7)

SOUTHWEST_MONSOON = (152, 273)
NORTHEAST_MONSOON = (274, 334)
PRE_MONSOON = (60, 151)
DRY_SEASON = [(1, 59), (335, 366)]


def get_season_factor(day_of_year: int) -> float:
    """
    Returns seasonal rainfall multiplier based on day of year.
    Southwest monsoon: 3.0-5.0x, NE monsoon: 1.0-2.0x, Pre-monsoon: 0.3-0.8x, Dry: 0.0-0.1x
    """
    if SOUTHWEST_MONSOON[0] <= day_of_year <= SOUTHWEST_MONSOON[1]:
        peak_offset = abs(day_of_year - 197)
        base = 3.0 + 2.0 * np.exp(-peak_offset ** 2 / (2 * 25 ** 2))
        return np.clip(base + np.random.normal(0, 0.3), 2.5, 5.5)
    elif NORTHEAST_MONSOON[0] <= day_of_year <= NORTHEAST_MONSOON[1]:
        return np.clip(np.random.uniform(1.0, 2.0) + np.random.normal(0, 0.2), 0.8, 2.5)
    elif PRE_MONSOON[0] <= day_of_year <= PRE_MONSOON[1]:
        return np.clip(np.random.uniform(0.3, 0.8) + np.random.normal(0, 0.1), 0.2, 1.0)
    else:
        return np.clip(np.random.uniform(0.0, 0.1) + np.random.normal(0, 0.05), 0.0, 0.2)


def generate_base_rainfall(day_of_year: int, year: int) -> float:
    """
    Generate base rainfall value with realistic distribution patterns.
    Uses gamma distribution for positive-skewed rainfall data.
    """
    season_factor = get_season_factor(day_of_year)
    
    if season_factor > 2.0:
        base = np.random.gamma(shape=2.5, scale=15 * season_factor)
        if np.random.random() < 0.02:
            base += np.random.exponential(scale=100)
    elif season_factor > 0.5:
        base = np.random.gamma(shape=1.8, scale=8 * season_factor)
    else:
        if np.random.random() < 0.7:
            return 0.0
        base = np.random.exponential(scale=2)
    
    return max(0, base)


def generate_temperature(day_of_year: int, temp_type: str) -> float:
    """
    Generate temperature with seasonal sinusoidal pattern.
    temp_type: 'max' or 'min'
    """
    annual_phase = 2 * np.pi * (day_of_year - 100) / 365
    seasonal_component = 8 * np.sin(annual_phase)
    
    if temp_type == 'max':
        base = np.mean(TEMP_MAX_RANGE) + seasonal_component
        noise = np.random.normal(0, 1.5)
        return np.clip(base + noise, *TEMP_MAX_RANGE)
    else:
        base = np.mean(TEMP_MIN_RANGE) + seasonal_component * 0.7
        noise = np.random.normal(0, 1.2)
        return np.clip(base + noise, *TEMP_MIN_RANGE)


def generate_humidity(day_of_year: int, rainfall_mm: float) -> float:
    """
    Generate humidity with seasonal pattern and rainfall correlation.
    Higher humidity during monsoon and on rainy days.
    """
    seasonal = 75 + 15 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
    rain_effect = 15 * (1 - np.exp(-rainfall_mm / 30))
    noise = np.random.normal(0, 4)
    value = seasonal + rain_effect + noise
    return np.clip(value, *HUMIDITY_RANGE)


def generate_cloud_cover(day_of_year: int, humidity: float, rainfall_mm: float) -> float:
    """
    Generate cloud cover correlated with humidity and rainfall.
    """
    seasonal = 50 + 30 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
    humidity_effect = max(0, (humidity - 60) * 0.5)
    rain_effect = min(30, rainfall_mm * 0.3)
    noise = np.random.normal(0, 8)
    value = seasonal + humidity_effect + rain_effect + noise
    return np.clip(value, *CLOUD_COVER_RANGE)


def generate_sunshine_hours(cloud_cover: float, day_of_year: int) -> float:
    """
    Generate sunshine hours inversely correlated with cloud cover.
    Also varies by season (shorter days in winter).
    """
    daylight = 12 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    cloud_reduction = cloud_cover * 0.1
    noise = np.random.normal(0, 0.8)
    value = max(0, daylight - cloud_reduction + noise)
    return np.clip(value, 0, min(13, daylight + 1))


def generate_wind_speed(day_of_year: int, rainfall_mm: float) -> float:
    """
    Generate wind speed with monsoon enhancement and storm correlation.
    """
    seasonal = 15 + 8 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
    rain_effect = min(12, rainfall_mm * 0.15)
    noise = np.random.normal(0, 3)
    value = seasonal + rain_effect + noise
    return np.clip(value, *WIND_SPEED_RANGE)


def generate_pressure(humidity: float, rainfall_mm: float) -> float:
    """
    Generate atmospheric pressure: lower during rainy/stormy conditions.
    """
    base = 1005
    humidity_effect = -0.03 * max(0, humidity - 75)
    rain_effect = -0.1 * min(20, rainfall_mm)
    noise = np.random.normal(0, 2)
    value = base + humidity_effect + rain_effect + noise
    return np.clip(value, *PRESSURE_RANGE)


def generate_dew_point(temperature: float, humidity: float) -> float:
    """
    Calculate dew point using Magnus approximation.
    Td = (b * α) / (a - α), where α = (a * T)/(b + T) + ln(RH/100)
    a=17.27, b=237.7°C for water vapor
    """
    a, b = 17.27, 237.7
    alpha = (a * temperature) / (b + temperature) + np.log(max(0.01, humidity) / 100)
    dew_point = (b * alpha) / (a - alpha)
    
    noise = np.random.normal(0, 0.5)
    return np.clip(dew_point + noise, *DEW_POINT_RANGE)


def generate_evapotranspiration(temperature: float, sunshine: float, 
                                humidity: float, wind: float) -> float:
    """
    Approximate reference evapotranspiration (simplified Penman-Monteith).
    Higher with temperature, sunshine, wind; lower with humidity.
    """
    temp_factor = (temperature + 17.8) * 0.0023
    radiation_factor = sunshine / 12
    humidity_factor = 1 - (humidity - 55) / 100
    wind_factor = 1 + wind / 50
    et = temp_factor * radiation_factor * humidity_factor * wind_factor * 10
    noise = np.random.normal(0, 0.3)
    return np.clip(et + noise, *EVAPOTRANSPIRATION_RANGE)


def generate_is_monsoon(day_of_year: int) -> int:
    if (SOUTHWEST_MONSOON[0] <= day_of_year <= SOUTHWEST_MONSOON[1] or 
        NORTHEAST_MONSOON[0] <= day_of_year <= NORTHEAST_MONSOON[1]):
        return 1
    return 0


def generate_date_range(start_year: int, end_year: int) -> pd.DatetimeIndex:
    return pd.date_range(start=f"{start_year}-01-01", 
                        end=f"{end_year}-12-31", 
                        freq='D')


def generate_dataset() -> pd.DataFrame:
    """
    Main function to generate the complete synthetic dataset.
    Returns DataFrame with all required columns.
    """
    print(f"Generating synthetic weather data for {REGION} ({START_YEAR}-{END_YEAR})...")
    
    dates = generate_date_range(START_YEAR, END_YEAR)
    n_records = len(dates)
    
    data = {
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'year': [d.year for d in dates],
        'month': [d.month for d in dates],
        'day': [d.day for d in dates],
        'day_of_year': [d.timetuple().tm_yday for d in dates],
    }
    print("Generating base features...")
    rainfall = np.array([generate_base_rainfall(doy, year) 
                        for doy, year in zip(data['day_of_year'], data['year'])])
    data['rainfall_mm'] = np.round(rainfall, 2)
    temp_max = np.array([generate_temperature(doy, 'max') for doy in data['day_of_year']])
    temp_min = np.array([generate_temperature(doy, 'min') for doy in data['day_of_year']])
    temp_min = np.minimum(temp_min, temp_max - 2)
    data['temperature_max'] = np.round(temp_max, 1)
    data['temperature_min'] = np.round(temp_min, 1)
    humidity = np.array([generate_humidity(doy, rain) 
                        for doy, rain in zip(data['day_of_year'], data['rainfall_mm'])])
    data['humidity'] = np.round(humidity, 1)
    cloud = np.array([generate_cloud_cover(doy, hum, rain) 
                     for doy, hum, rain in zip(data['day_of_year'], 
                                              data['humidity'], 
                                              data['rainfall_mm'])])
    data['cloud_cover'] = np.round(cloud, 1)
    sunshine = np.array([generate_sunshine_hours(cc, doy) 
                        for cc, doy in zip(data['cloud_cover'], data['day_of_year'])])
    data['sunshine_hours'] = np.round(sunshine, 1)
    wind = np.array([generate_wind_speed(doy, rain) 
                    for doy, rain in zip(data['day_of_year'], data['rainfall_mm'])])
    data['wind_speed'] = np.round(wind, 1)
    pressure = np.array([generate_pressure(hum, rain) 
                        for hum, rain in zip(data['humidity'], data['rainfall_mm'])])
    data['pressure'] = np.round(pressure, 1)
    dew = np.array([generate_dew_point(tmax, hum) 
                   for tmax, hum in zip(data['temperature_max'], data['humidity'])])
    data['dew_point'] = np.round(dew, 1)
    et = np.array([generate_evapotranspiration(tmax, sun, hum, wind) 
                  for tmax, sun, hum, wind in zip(data['temperature_max'],
                                                  data['sunshine_hours'],
                                                  data['humidity'],
                                                  data['wind_speed'])])
    data['evapotranspiration'] = np.round(et, 2)
    data['is_monsoon'] = [generate_is_monsoon(doy) for doy in data['day_of_year']]
    df = pd.DataFrame(data)
    df['previous_7day_rainfall'] = df['rainfall_mm'].rolling(window=7, min_periods=1).mean().round(2)
    df['previous_30day_rainfall'] = df['rainfall_mm'].rolling(window=30, min_periods=1).mean().round(2)
    df = df.fillna(0)
    numeric_cols = ['temperature_max', 'temperature_min', 'humidity', 'wind_speed', 
                   'pressure', 'cloud_cover', 'sunshine_hours', 'dew_point',
                   'evapotranspiration', 'rainfall_mm', 'previous_7day_rainfall', 
                   'previous_30day_rainfall']
    for col in numeric_cols:
        df[col] = df[col].clip(lower=0)
    print(f"Generated {len(df):,} daily records")
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Perform data quality validation and return report.
    """
    report = {
        'total_records': len(df),
        'date_range': f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}",
        'columns': list(df.columns),
        'null_counts': df.isnull().sum().to_dict(),
        'negative_values': {},
        'rainfall_stats': {
            'annual_totals': df.groupby('year')['rainfall_mm'].sum().describe().to_dict(),
            'monthly_means': df.groupby('month')['rainfall_mm'].mean().round(1).to_dict(),
            'monsoon_vs_dry': {
                'monsoon_mean': df[df['is_monsoon'] == 1]['rainfall_mm'].mean(),
                'dry_mean': df[df['is_monsoon'] == 0]['rainfall_mm'].mean(),
            }
        },
        'feature_ranges': {}
    }
    
    numeric_cols = ['temperature_max', 'temperature_min', 'humidity', 'wind_speed', 
                   'pressure', 'cloud_cover', 'sunshine_hours', 'dew_point',
                   'evapotranspiration', 'rainfall_mm']
    for col in numeric_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            report['negative_values'][col] = int(neg_count)
    for col in numeric_cols:
        report['feature_ranges'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean())
        }
    
    return report


def print_quality_report(report: dict):
    """Print formatted quality report to console."""
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)
    print(f"Total Records: {report['total_records']:,}")
    print(f"Date Range: {report['date_range']}")
    print(f"Columns ({len(report['columns'])}): {', '.join(report['columns'])}")
    
    nulls = {k: v for k, v in report['null_counts'].items() if v > 0}
    if nulls:
        print(f"NULL VALUES FOUND: {nulls}")
    else:
        print("No NULL values in any column")
    if report['negative_values']:
        print(f"NEGATIVE VALUES FOUND: {report['negative_values']}")
    else:
        print("No negative values in numeric columns")
    
    print("\nRAINFALL STATISTICS:")
    annual = report['rainfall_stats']['annual_totals']
    print(f"   Annual Total - Mean: {annual['mean']:.0f}mm, Std: {annual['std']:.0f}mm")
    print(f"   Annual Total - Range: {annual['min']:.0f}mm to {annual['max']:.0f}mm")
    monthly = report['rainfall_stats']['monthly_means']
    print(f"\n   Monthly Mean Rainfall (mm):")
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for m in range(1, 13):
        bar = '█' * int(monthly[m] / 10)
        print(f"   {month_names[m-1]:3s}: {monthly[m]:5.1f}mm {bar}")
    monsoon_stats = report['rainfall_stats']['monsoon_vs_dry']
    print(f"\n   Monsoon vs Dry Season:")
    print(f"   - Monsoon days avg: {monsoon_stats['monsoon_mean']:.1f}mm/day")
    print(f"   - Dry days avg: {monsoon_stats['dry_mean']:.1f}mm/day")
    
    print("\nFEATURE RANGES (sample):")
    sample_features = ['temperature_max', 'humidity', 'cloud_cover', 'rainfall_mm']
    for feat in sample_features:
        r = report['feature_ranges'][feat]
        print(f"   {feat:20s}: {r['min']:6.1f} - {r['max']:6.1f} (mean: {r['mean']:.1f})")
    
    print("="*70 + "\n")


def save_dataset(df: pd.DataFrame, output_path: str = "data/dakshina_kannada_weather.csv"):
    """Save dataset to CSV with proper formatting."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    """Main execution function."""
    start_time = datetime.now()
    
    df = generate_dataset()
    report = validate_dataset(df)
    print_quality_report(report)
    save_dataset(df)
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Generation completed in {elapsed:.1f} seconds")
    print("Ready for model training. Next: python train_models.py")
    
    return df, report


if __name__ == "__main__":
    df, report = main()