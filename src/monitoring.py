# src/monitoring.py
import pandas as pd
import numpy as np
import pickle
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from scipy.spatial.distance import jensenshannon

class DriftDetector:
    """
    Detects distribution drift in feature data by comparing new data
    against a reference dataset (training data).
    """
    
    def __init__(self, reference_data=None, feature_names=None, p_threshold=0.05):
        """
        Initialize the drift detector.
        
        Args:
            reference_data: DataFrame or path to reference data
            feature_names: List of feature names to monitor
            p_threshold: p-value threshold for statistical tests
        """
        self.p_threshold = p_threshold
        self.reference_statistics = {}
        self.feature_names = feature_names
        self.drift_history = []
        
        # Load reference data if provided
        if reference_data is not None:
            self.set_reference_data(reference_data)
    
    def set_reference_data(self, reference_data):
        """Set reference data and compute baseline statistics"""
        # Handle file path or direct DataFrame
        if isinstance(reference_data, str) and os.path.exists(reference_data):
            self.reference_data = pd.read_csv(reference_data)
        else:
            self.reference_data = reference_data
        
        # If feature names not provided, use all numeric columns
        if self.feature_names is None:
            self.feature_names = self.reference_data.select_dtypes(include=np.number).columns.tolist()
        
        # Compute reference statistics
        self._compute_reference_statistics()
        
        # Create monitoring directory if it doesn't exist
        os.makedirs('monitoring/drift', exist_ok=True)
        
        # Save reference statistics
        with open('monitoring/drift/reference_statistics.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            stats_dict = {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                             for kk, vv in v.items()} 
                         for k, v in self.reference_statistics.items()}
            json.dump(stats_dict, f, indent=2)
    
    def _compute_reference_statistics(self):
        """Compute baseline statistics for reference data"""
        for feature in self.feature_names:
            feature_data = self.reference_data[feature]
            self.reference_statistics[feature] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'median': np.median(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'q1': np.percentile(feature_data, 25),
                'q3': np.percentile(feature_data, 75),
                'hist': np.histogram(feature_data, bins=10, density=True)[0].tolist()
            }
    
    def check_drift(self, current_data, log=True, plot=False):
        """
        Check for drift between current data and reference data.
        
        Args:
            current_data: DataFrame of current data to check for drift
            log: Whether to log drift results
            plot: Whether to generate and save comparison plots
            
        Returns:
            Dictionary with drift results
        """
        if isinstance(current_data, str) and os.path.exists(current_data):
            current_data = pd.read_csv(current_data)
        
        drift_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': {},
            'drift_detected': False
        }
        
        for feature in self.feature_names:
            if feature not in current_data.columns:
                continue
                
            ref_data = self.reference_data[feature]
            curr_data = current_data[feature]
            
            # Statistical tests
            ks_stat, ks_pvalue = stats.ks_2samp(ref_data, curr_data)
            
            # Distribution distance
            ref_hist = np.histogram(ref_data, bins=10, density=True)[0]
            curr_hist = np.histogram(curr_data, bins=10, density=True)[0]
            # Add small epsilon to avoid division by zero
            ref_hist = ref_hist + 1e-10
            curr_hist = curr_hist + 1e-10
            # Normalize
            ref_hist = ref_hist / np.sum(ref_hist)
            curr_hist = curr_hist / np.sum(curr_hist)
            js_distance = jensenshannon(ref_hist, curr_hist)
            
            # Basic statistics comparison
            mean_diff = abs(np.mean(curr_data) - self.reference_statistics[feature]['mean'])
            std_diff = abs(np.std(curr_data) - self.reference_statistics[feature]['std'])
            
            # Determine if drift occurred
            is_drift = ks_pvalue < self.p_threshold
            
            drift_results['features'][feature] = {
                'ks_pvalue': float(ks_pvalue),
                'ks_statistic': float(ks_stat),
                'js_distance': float(js_distance),
                'mean_difference': float(mean_diff),
                'std_difference': float(std_diff),
                'is_drift': is_drift
            }
            
            if is_drift:
                drift_results['drift_detected'] = True
            
            # Generate plots if requested
            if plot:
                self._generate_comparison_plot(feature, ref_data, curr_data, is_drift)
        
        # Log results if requested
        if log:
            self.drift_history.append(drift_results)
            self._log_drift_results(drift_results)
        
        return drift_results
    
    def _generate_comparison_plot(self, feature, ref_data, curr_data, is_drift):
        """Generate comparison plots between reference and current data"""
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram comparison
        sns.histplot(ref_data, color='blue', label='Reference', alpha=0.5, ax=ax[0])
        sns.histplot(curr_data, color='red', label='Current', alpha=0.5, ax=ax[0])
        ax[0].set_title(f'{feature} Distribution Comparison')
        ax[0].legend()
        
        # Box plot comparison
        box_data = pd.DataFrame({
            'Reference': ref_data,
            'Current': curr_data
        })
        sns.boxplot(data=box_data, ax=ax[1])
        ax[1].set_title(f'{feature} Box Plot Comparison')
        
        # Add drift indication to title
        drift_status = "DRIFT DETECTED" if is_drift else "No Drift"
        fig.suptitle(f'{feature} - {drift_status}', fontsize=16, color='red' if is_drift else 'green')
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_dir = f'monitoring/drift/plots/{timestamp}'
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f'{plot_dir}/{feature}_comparison.png')
        plt.close()
    
    def _log_drift_results(self, drift_results):
        """Log drift results to a file"""
        log_dir = 'monitoring/drift/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = f'{log_dir}/drift_log.jsonl'
        
        # Convert numpy values to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        converted_results = json.loads(
            json.dumps(drift_results, default=convert_numpy)
        )
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(converted_results) + '\n')

def setup_drift_monitoring(processed_data_path='data/processed/processed_iris.csv'):
    """Set up drift monitoring using processed training data as reference"""
    # Create monitoring directory structure
    os.makedirs('monitoring/drift/logs', exist_ok=True)
    os.makedirs('monitoring/drift/plots', exist_ok=True)
    
    # Initialize drift detector with training data
    if os.path.exists(processed_data_path):
        reference_data = pd.read_csv(processed_data_path)
        # Use only feature columns (exclude target)
        feature_names = [col for col in reference_data.columns if col != 'target']
        reference_features = reference_data[feature_names]
        
        detector = DriftDetector(reference_data=reference_features, 
                                feature_names=feature_names)
        
        # Save the detector for future use
        os.makedirs('models', exist_ok=True)
        with open('models/drift_detector.pkl', 'wb') as f:
            pickle.dump(detector, f)
        
        print(f"Drift detector initialized with reference data from {processed_data_path}")
        print(f"Monitoring set up for features: {feature_names}")
        return detector
    else:
        print(f"Error: Reference data file {processed_data_path} not found")
        return None

def get_drift_detector():
    """Get the drift detector object, creating it if necessary"""
    detector_path = 'models/drift_detector.pkl'
    
    if os.path.exists(detector_path):
        with open(detector_path, 'rb') as f:
            detector = pickle.load(f)
        return detector
    else:
        return setup_drift_monitoring()