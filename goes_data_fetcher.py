#!/usr/bin/env python3
"""
GOES X-ray Data Fetcher

This module provides functions to fetch real-time and historical GOES X-ray flux data
for solar flare monitoring. It interfaces with NOAA's Space Weather APIs and provides
processed data in a format suitable for the solar flare monitoring system.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)

class GOESDataFetcher:
    """Fetches and processes GOES X-ray flux data"""
    
    def __init__(self):
        self.base_url = "https://services.swpc.noaa.gov"
        self.session = None
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_realtime_xray_flux(self) -> Dict:
        """Get real-time GOES X-ray flux data"""
        try:
            url = f"{self.base_url}/products/summary/solar-wind-speed.json"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self.process_xray_data(data)
                else:
                    logger.warning(f"GOES API returned status {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Error fetching GOES X-ray data: {e}")
            return {}
    
    async def get_historical_xray_flux(self, hours: int = 24) -> Dict:
        """Get historical GOES X-ray flux data"""
        try:
            # Use 24-hour historical data endpoint
            url = f"{self.base_url}/products/solar-wind/xrays-24-hour.json"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self.process_historical_data(data, hours)
                else:
                    logger.warning(f"GOES historical API returned status {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Error fetching GOES historical data: {e}")
            return {}
    
    def process_xray_data(self, raw_data: Dict) -> Dict:
        """Process raw GOES X-ray data into structured format"""
        try:
            current_time = datetime.utcnow()
            
            # Extract relevant fields (adjust based on actual API response structure)
            processed_data = {
                'timestamp': current_time.isoformat(),
                'short_wavelength': self.extract_flux_value(raw_data, 'short'),
                'long_wavelength': self.extract_flux_value(raw_data, 'long'),
                'flare_class': self.determine_flare_class(raw_data),
                'background_level': self.get_background_level(raw_data),
                'status': 'active',
                'source': 'GOES-16/18'
            }
            
            # Add derived metrics
            processed_data['peak_flux'] = max(
                processed_data.get('short_wavelength', 0),
                processed_data.get('long_wavelength', 0)
            )
            
            processed_data['activity_level'] = self.classify_activity_level(
                processed_data['peak_flux']
            )
            
            return processed_data
        
        except Exception as e:
            logger.error(f"Error processing GOES X-ray data: {e}")
            return {}
    
    def process_historical_data(self, raw_data: List, hours: int) -> Dict:
        """Process historical GOES data"""
        try:
            if not raw_data:
                return {}
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(raw_data)
            
            # Filter to requested time window
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Process timestamps and filter
            if 'time_tag' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time_tag'])
                df = df[df['timestamp'] >= start_time]
            
            # Extract flux data
            flux_columns = [col for col in df.columns if 'flux' in col.lower()]
            
            processed_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'data_points': len(df),
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'flux_data': df[flux_columns].to_dict('records') if flux_columns else [],
                'statistics': self.calculate_flux_statistics(df, flux_columns),
                'flare_events': self.detect_flare_events(df, flux_columns),
                'status': 'historical'
            }
            
            return processed_data
        
        except Exception as e:
            logger.error(f"Error processing historical GOES data: {e}")
            return {}
    
    def extract_flux_value(self, data: Dict, wavelength: str) -> float:
        """Extract flux value for specific wavelength"""
        try:
            # Adjust field names based on actual API response
            flux_fields = {
                'short': ['short_channel', 'xrs_a', '0.5-4.0'],
                'long': ['long_channel', 'xrs_b', '1.0-8.0']
            }
            
            for field in flux_fields.get(wavelength, []):
                if field in data:
                    return float(data[field])
            
            return 0.0
        
        except Exception:
            return 0.0
    
    def determine_flare_class(self, data: Dict) -> str:
        """Determine flare classification from GOES data"""
        try:
            # Get peak flux (usually from long wavelength channel)
            peak_flux = self.extract_flux_value(data, 'long')
            
            if peak_flux >= 1e-3:  # X-class
                magnitude = peak_flux / 1e-4
                return f"X{magnitude:.1f}"
            elif peak_flux >= 1e-4:  # M-class
                magnitude = peak_flux / 1e-5
                return f"M{magnitude:.1f}"
            elif peak_flux >= 1e-5:  # C-class
                magnitude = peak_flux / 1e-6
                return f"C{magnitude:.1f}"
            elif peak_flux >= 1e-6:  # B-class
                magnitude = peak_flux / 1e-7
                return f"B{magnitude:.1f}"
            else:  # A-class or background
                magnitude = peak_flux / 1e-8
                return f"A{magnitude:.1f}"
        
        except Exception:
            return "Unknown"
    
    def get_background_level(self, data: Dict) -> float:
        """Get background X-ray level"""
        try:
            # Background is typically the minimum flux over recent period
            return self.extract_flux_value(data, 'long')
        except Exception:
            return 0.0
    
    def classify_activity_level(self, peak_flux: float) -> str:
        """Classify solar activity level based on peak flux"""
        if peak_flux >= 1e-3:
            return "Extreme"
        elif peak_flux >= 1e-4:
            return "High"
        elif peak_flux >= 1e-5:
            return "Moderate"
        elif peak_flux >= 1e-6:
            return "Low"
        else:
            return "Quiet"
    
    def calculate_flux_statistics(self, df: pd.DataFrame, flux_columns: List[str]) -> Dict:
        """Calculate statistical measures of X-ray flux"""
        try:
            stats = {}
            
            for col in flux_columns:
                if col in df.columns:
                    values = pd.to_numeric(df[col], errors='coerce').dropna()
                    
                    if len(values) > 0:
                        stats[col] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'median': float(values.median()),
                            'p95': float(values.quantile(0.95)),
                            'count': len(values)
                        }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error calculating flux statistics: {e}")
            return {}
    
    def detect_flare_events(self, df: pd.DataFrame, flux_columns: List[str]) -> List[Dict]:
        """Detect flare events in historical data"""
        try:
            events = []
            
            for col in flux_columns:
                if col in df.columns:
                    values = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    # Simple flare detection: values above threshold
                    threshold = values.quantile(0.95)  # Top 5% as potential flares
                    
                    flare_indices = values[values > threshold].index
                    
                    for idx in flare_indices:
                        if 'timestamp' in df.columns:
                            timestamp = df.loc[idx, 'timestamp']
                        else:
                            timestamp = datetime.utcnow() - timedelta(hours=24-idx/len(df)*24)
                        
                        events.append({
                            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                            'channel': col,
                            'flux': float(values.iloc[idx]),
                            'class': self.determine_flare_class({'long_channel': values.iloc[idx]}),
                            'duration_estimate': 10  # minutes, rough estimate
                        })
            
            # Sort by timestamp
            events.sort(key=lambda x: x['timestamp'])
            
            return events
        
        except Exception as e:
            logger.error(f"Error detecting flare events: {e}")
            return []

async def test_goes_fetcher():
    """Test function for GOES data fetcher"""
    async with GOESDataFetcher() as goes:
        print("Testing GOES X-ray data fetcher...")
        
        # Test real-time data
        realtime_data = await goes.get_realtime_xray_flux()
        print(f"Real-time data: {realtime_data}")
        
        # Test historical data
        historical_data = await goes.get_historical_xray_flux(hours=6)
        print(f"Historical data points: {historical_data.get('data_points', 0)}")
        
        return realtime_data, historical_data

if __name__ == "__main__":
    asyncio.run(test_goes_fetcher())
