"""
Setup script for the ML Recommendation System
Handles installation, data generation, model training, and API startup
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install Python requirements"""
    if not os.path.exists('requirements.txt'):
        print("Error: requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python requirements"
    )

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")
    
    return True

def generate_synthetic_data(n_samples=2000):
    """Generate synthetic dataset"""
    print(f"\nGenerating synthetic dataset with {n_samples} samples...")
    
    try:
        from data_generator import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(n_samples=n_samples)
        df = generator.save_dataset('data')
        
        print(f"✓ Generated {len(df)} samples")
        print(f"✓ Dataset saved to data/synthetic_student_data.csv")
        return True
        
    except Exception as e:
        print(f"✗ Failed to generate synthetic data: {str(e)}")
        return False

def train_models():
    """Train ML models"""
    print("\nTraining ML models...")
    
    try:
        from strand_recommender import StrandRecommendationSystem
        
        # Check if data exists
        data_file = 'data/synthetic_student_data.csv'
        if not os.path.exists(data_file):
            print("Data file not found. Generating synthetic data first...")
            if not generate_synthetic_data():
                return False
        
        # Initialize and train
        recommender = StrandRecommendationSystem(data_file)
        X, y_track, y_strand = recommender.load_data()
        results = recommender.train_models(X, y_track, y_strand)
        recommender.save_models('models')
        
        print(f"✓ Track model accuracy: {results['track_accuracy']:.4f}")
        print(f"✓ Strand model accuracy: {results['strand_accuracy']:.4f}")
        print("✓ Models saved to models/ directory")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to train models: {str(e)}")
        return False

def test_api():
    """Test the API with a sample request"""
    print("\nTesting API...")
    
    try:
        import requests
        import json
        import time
        
        # Wait a moment for API to start
        time.sleep(2)
        
        # Test health endpoint
        response = requests.get('http://localhost:5000/health', timeout=10)
        if response.status_code == 200:
            print("✓ API health check passed")
        else:
            print(f"✗ API health check failed: {response.status_code}")
            return False
        
        # Test sample request
        sample_response = requests.get('http://localhost:5000/sample-request', timeout=10)
        if sample_response.status_code == 200:
            sample_data = sample_response.json()['sample_request']
            
            # Test recommendation endpoint
            rec_response = requests.post('http://localhost:5000/recommend', 
                                       json=sample_data, timeout=30)
            
            if rec_response.status_code == 200:
                recommendations = rec_response.json()
                print(f"✓ API test passed - got {len(recommendations.get('recommendations', []))} recommendations")
                return True
            else:
                print(f"✗ API recommendation test failed: {rec_response.status_code}")
                print(rec_response.text)
                return False
        else:
            print(f"✗ Failed to get sample request: {sample_response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ API test failed: {str(e)}")
        return False

def start_api(test_mode=False):
    """Start the Flask API"""
    print("\nStarting Flask API...")
    
    if test_mode:
        # Start API in background for testing
        import threading
        import time
        
        def run_api():
            try:
                from api import app, initialize_system
                if initialize_system():
                    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
            except Exception as e:
                print(f"API startup error: {e}")
        
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # Wait for API to start
        time.sleep(5)
        
        return test_api()
    else:
        # Start API normally
        try:
            from api import app, initialize_system
            if initialize_system():
                print("✓ ML system initialized successfully")
                print("✓ Starting Flask API on http://localhost:5000")
                print("\nAPI Endpoints:")
                print("  GET  /health              - Health check")
                print("  POST /recommend           - Get recommendations")
                print("  GET  /feature-importance  - Get feature importance")
                print("  POST /retrain             - Retrain models")
                print("  GET  /sample-request      - Get sample request format")
                print("\nPress Ctrl+C to stop the server")
                
                app.run(debug=True, host='0.0.0.0', port=5000)
                return True
            else:
                print("✗ Failed to initialize ML system")
                return False
        except KeyboardInterrupt:
            print("\n✓ API server stopped")
            return True
        except Exception as e:
            print(f"✗ Failed to start API: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='ML Recommendation System Setup')
    parser.add_argument('--action', choices=['install', 'generate-data', 'train', 'start-api', 'full-setup', 'test'], 
                       default='full-setup', help='Action to perform')
    parser.add_argument('--samples', type=int, default=2000, help='Number of synthetic samples to generate')
    parser.add_argument('--skip-install', action='store_true', help='Skip package installation')
    
    args = parser.parse_args()
    
    print("=== ML Recommendation System Setup ===")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    success = True
    
    if args.action == 'install' or (args.action == 'full-setup' and not args.skip_install):
        success = success and install_requirements()
    
    if args.action == 'generate-data' or args.action == 'full-setup':
        success = success and create_directories()
        success = success and generate_synthetic_data(args.samples)
    
    if args.action == 'train' or args.action == 'full-setup':
        success = success and train_models()
    
    if args.action == 'start-api' or args.action == 'full-setup':
        if success:
            start_api(test_mode=False)
        else:
            print("\n✗ Setup failed. Cannot start API.")
            sys.exit(1)
    
    if args.action == 'test':
        success = success and create_directories()
        if not os.path.exists('data/synthetic_student_data.csv'):
            success = success and generate_synthetic_data(args.samples)
        if not os.path.exists('models/track_model.pkl'):
            success = success and train_models()
        if success:
            success = start_api(test_mode=True)
    
    if success and args.action != 'start-api':
        print("\n✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the API: python setup.py --action start-api")
        print("2. Update your Laravel .env file with: ML_API_URL=http://localhost:5000")
        print("3. Test the integration from your Laravel application")
    elif not success:
        print("\n✗ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()