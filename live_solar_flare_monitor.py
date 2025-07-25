import os
import sys
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import drms
from dotenv import load_dotenv

# Load project modules
sys.path.append('Running Models')
sys.path.append('Downloading Data')
sys.path.append('util')
sys.path.append('.')

try:
    from ensemble_predict import EnsemblePredictor
    from get_jsoc_data import download_sharp_time_series
    from goes_data_fetcher import GOESDataFetcher
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleSolarFlareMonitor:
    
    def __init__(self):
        """Initialize the monitoring system"""
        self.running = False
        
        # Initialize DRMS client
        self.jsoc_email = os.getenv("EMAIL")
        if not self.jsoc_email:
            logger.error("EMAIL environment variable not set in .env file")
            raise ValueError("EMAIL is required for JSOC access")
        
        self.drms_client = drms.Client(email=self.jsoc_email)
        logger.info(f"DRMS client initialized with email: {self.jsoc_email}")
        
        # Data directory
        self.data_dir = Path("live_monitor_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Load ensemble predictor
        try:
            self.ensemble = EnsemblePredictor()
            logger.info("Ensemble predictor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ensemble predictor: {e}")
            self.ensemble = None
    
    async def get_active_regions(self) -> List[int]:
        try:
            # Query for recent SHARP data to find active regions
            current_time = datetime.utcnow()
            
            # Look for data from last 12 hours
            for hours_back in range(0, 12, 2):
                query_time = current_time - timedelta(hours=hours_back)
                date_str = query_time.strftime('%Y.%m.%d_%H:%M:%S_TAI')
                
                # Query for any SHARP data at this time
                query = f"hmi.sharp_cea_720s[][{date_str}]"
                
                try:
                    result = self.drms_client.query(query, key=["HARPNUM"])
                    if result is not None and not result.empty:
                        harp_numbers = result['HARPNUM'].unique().tolist()
                        # Filter to reasonable HARP numbers (> 1000)
                        harp_numbers = [h for h in harp_numbers if h > 1000]
                        if harp_numbers:
                            logger.info(f"Found {len(harp_numbers)} active regions: {harp_numbers[:5]}...")
                            return harp_numbers[:3]  # Return max 3 for simplicity
                except Exception as e:
                    logger.debug(f"No data found for {date_str}: {e}")
                    continue
            
            logger.warning("No active regions found")
            return []
            
        except Exception as e:
            logger.error(f"Error getting active regions: {e}")
            return []
    
    async def download_hmi_data(self, harpnum: int) -> Optional[np.ndarray]:
        """Download and process HMI data for prediction"""
        try:
            # Get data for last 6 hours (6 timesteps, 1 hour apart)
            end_time = datetime.utcnow()
            time_points = []
            
            for i in range(6):
                time_point = end_time - timedelta(hours=i)
                time_points.append(time_point.strftime('%Y-%m-%d %H:%M:%S'))
            
            time_points.reverse()  # Chronological order
            
            # Create output directory
            output_dir = self.data_dir / f"harp_{harpnum}_{end_time.strftime('%Y%m%d_%H%M')}"
            
            # Download data
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, 
                download_sharp_time_series, 
                harpnum, 
                time_points,
                str(output_dir)
            )
            
            if success:
                # Process data for model input
                processed_data = await self.process_data(output_dir)
                return processed_data
            else:
                logger.warning(f"Failed to download data for HARP {harpnum}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading HMI data for HARP {harpnum}: {e}")
            return None
    
    async def process_data(self, data_dir: Path) -> Optional[np.ndarray]:
        """Process downloaded FITS files into model input"""
        try:
            from astropy.io import fits
            from skimage.transform import resize
            
            # Find timestep directories
            timestep_dirs = sorted([d for d in data_dir.iterdir() 
                                  if d.is_dir() and d.name.startswith('timestep_')])
            
            if len(timestep_dirs) == 0:
                return None
            
            channels = ['Bp', 'Bt', 'Br', 'continuum']
            processed_timesteps = []
            
            for timestep_dir in timestep_dirs:
                timestep_data = np.zeros((128, 128, 4), dtype=np.float32)
                
                for i, channel in enumerate(channels):
                    fits_files = list(timestep_dir.glob(f"*{channel}*.fits"))
                    if fits_files:
                        with fits.open(fits_files[0]) as hdul:
                            data = hdul[1].data
                            if data is not None:
                                # Resize to 128x128
                                resized = resize(data, (128, 128), preserve_range=True)
                                # Simple normalization
                                resized = np.nan_to_num(resized)
                                if np.max(np.abs(resized)) > 0:
                                    resized = resized / np.max(np.abs(resized))
                                timestep_data[:, :, i] = resized
                
                processed_timesteps.append(timestep_data)
            
            # Stack into (timesteps, height, width, channels)
            if processed_timesteps:
                model_input = np.stack(processed_timesteps, axis=0)
                logger.info(f"Processed data shape: {model_input.shape}")
                return model_input
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
    
    async def make_prediction(self, data: np.ndarray, harpnum: int) -> Dict:
        """Make flare prediction using ensemble model"""
        if self.ensemble is None:
            return {}
        
        try:
            # Add batch dimension
            batch_data = np.expand_dims(data, axis=0)
            
            # Make prediction
            ensemble_pred, individual_preds = self.ensemble.predict_batch(batch_data)
            
            result = {
                'harpnum': harpnum,
                'timestamp': datetime.utcnow().isoformat(),
                'ensemble_probability': float(ensemble_pred[0]),
                'prediction': 'FLARE' if ensemble_pred[0] > 0.5 else 'NO FLARE',
                'individual_predictions': {k: float(v[0]) for k, v in individual_preds.items()}
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {}
    
    async def get_goes_data(self) -> Dict:
        """Get current GOES X-ray flux"""
        try:
            async with GOESDataFetcher() as goes:
                data = await goes.get_realtime_xray_flux()
                return data
        except Exception as e:
            logger.error(f"Error getting GOES data: {e}")
            return {}
    
    def log_results(self, predictions: List[Dict], goes_data: Dict):
        """Log monitoring results"""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        print(f"\n{'='*60}")
        print(f"Solar Flare Monitor Update - {timestamp}")
        print(f"{'='*60}")
        
        if predictions:
            print(f"Active Regions Monitored: {len(predictions)}")
            for pred in predictions:
                harp = pred['harpnum']
                prob = pred['ensemble_probability']
                prediction = pred['prediction']
                print(f"  HARP {harp}: {prob:.1%} - {prediction}")
                
                # Check for high probability
                if prob > 0.7:
                    print(f"HIGH FLARE PROBABILITY")
        else:
            print("No active regions found or processed")
        
        if goes_data:
            print(f"GOES Status: Data available")
        else:
            print(f"GOES Status: No data")
        
        print(f"{'='*60}\n")
    
    async def run_cycle(self):
        """Run one monitoring cycle"""
        logger.info("Starting monitoring cycle...")
        
        # Get active regions
        active_regions = await self.get_active_regions()
        
        predictions = []
        
        # Process each active region
        for harpnum in active_regions:
            logger.info(f"Processing HARP {harpnum}...")
            
            # Download and process HMI data
            hmi_data = await self.download_hmi_data(harpnum)
            
            if hmi_data is not None:
                # Make prediction
                prediction = await self.make_prediction(hmi_data, harpnum)
                if prediction:
                    predictions.append(prediction)
        
        # Get GOES data
        goes_data = await self.get_goes_data()
        
        # Log results
        self.log_results(predictions, goes_data)
        
        return predictions, goes_data
    
    async def start_monitoring(self, interval_minutes: int = 30):
        """Start monitoring"""
        self.running = True
        logger.info(f"Starting solar flare monitoring (update every {interval_minutes} minutes)")
        
        try:
            while self.running:
                try:
                    await self.run_cycle()
                    
                    # Wait for next cycle
                    await asyncio.sleep(interval_minutes * 60)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
                    
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            self.running = False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False

async def main():
    print("Simple Solar Flare Monitor")
    print("============================")
    
    try:
        monitor = SimpleSolarFlareMonitor()
        await monitor.start_monitoring(interval_minutes=30)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
