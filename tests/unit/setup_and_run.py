#!/usr/bin/env python3
"""
Setup and run script for AI Scam Detection System.
This script helps set up the environment and run various components.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def run_command(command, description):
    """Run a shell command with description."""
    print(f"➤ {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def setup_environment():
    """Set up the development environment."""
    print_header("Setting Up Environment")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Create virtual environment
    if not Path("venv").exists():
        run_command("python -m venv venv", "Creating virtual environment")
    else:
        print("✓ Virtual environment already exists")
    
    # Activate instructions
    if os.name == 'nt':
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    print(f"\n📝 To activate virtual environment, run: {activate_cmd}")
    
    # Install dependencies
    print_header("Installing Dependencies")
    run_command("pip install --upgrade pip", "Upgrading pip")
    run_command("pip install -r requirements.txt", "Installing Python packages")
    
    # Download NLTK data
    print_header("Downloading NLTK Data")
    run_command(
        "python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')\"",
        "Downloading NLTK resources"
    )
    
    # Create necessary directories
    print_header("Creating Directories")
    directories = [
        'logs',
        'models/saved_models',
        'data/raw',
        'data/processed',
        'data/external'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")
    
    print("\n✅ Environment setup complete!")


def train_model():
    """Train the scam detection model."""
    print_header("Training Model")
    
    # Check if data exists
    data_path = Path("data/raw/scam_dataset.csv")
    if not data_path.exists():
        print(f"❌ Training data not found at {data_path}")
        print("Please ensure the dataset is available")
        return False
    
    # Run training
    success = run_command(
        "python src/training/train_model.py",
        "Training scam detection model"
    )
    
    if success:
        print("\n✅ Model training complete!")
        print("Model saved to: models/saved_models/scam_classifier.pkl")
    
    return success


def run_api():
    """Start the API server."""
    print_header("Starting API Server")
    
    # Check if model exists
    model_path = Path("models/saved_models/scam_classifier.pkl")
    if not model_path.exists():
        print("⚠️  Model not found. Training model first...")
        if not train_model():
            print("❌ Cannot start API without trained model")
            return
    
    print("Starting Flask API server...")
    print("API will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run("python api/app.py", shell=True)
    except KeyboardInterrupt:
        print("\n\n👋 API server stopped")


def run_tests():
    """Run unit tests."""
    print_header("Running Tests")
    
    success = run_command(
        "pytest tests/ -v --tb=short",
        "Running test suite"
    )
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")


def open_frontend():
    """Open the frontend in browser."""
    print_header("Opening Frontend")
    
    frontend_path = Path("frontend/public/index.html").absolute()
    
    if not frontend_path.exists():
        print(f"❌ Frontend not found at {frontend_path}")
        return
    
    print(f"Opening {frontend_path} in browser...")
    
    import webbrowser
    webbrowser.open(f"file://{frontend_path}")
    
    print("\n✅ Frontend opened!")
    print("Make sure the API server is running for full functionality")


def show_menu():
    """Show interactive menu."""
    while True:
        print("\n" + "=" * 60)
        print("  AI SCAM DETECTION SYSTEM - Main Menu")
        print("=" * 60)
        print("\n1. Setup Environment")
        print("2. Train Model")
        print("3. Run API Server")
        print("4. Open Frontend")
        print("5. Run Tests")
        print("6. Do Everything (Setup → Train → Run)")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == '1':
            setup_environment()
        elif choice == '2':
            train_model()
        elif choice == '3':
            run_api()
        elif choice == '4':
            open_frontend()
        elif choice == '5':
            run_tests()
        elif choice == '6':
            setup_environment()
            if train_model():
                print("\n🚀 Starting API server in 3 seconds...")
                import time
                time.sleep(3)
                run_api()
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please try again.")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'setup':
            setup_environment()
        elif command == 'train':
            train_model()
        elif command == 'api':
            run_api()
        elif command == 'frontend':
            open_frontend()
        elif command == 'test':
            run_tests()
        elif command == 'all':
            setup_environment()
            if train_model():
                run_api()
        else:
            print(f"Unknown command: {command}")
            print("\nUsage: python setup_and_run.py [setup|train|api|frontend|test|all]")
    else:
        show_menu()


if __name__ == '__main__':
    main()