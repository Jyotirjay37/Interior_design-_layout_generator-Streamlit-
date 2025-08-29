import os
import requests
from tqdm import tqdm

def download_yolov8_model():
    """Download YOLOv8 nano model from official Ultralytics repository"""
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    model_path = "yolov8n.pt"
    
    print("Downloading YOLOv8n model...")
    
    try:
        # Download the model with progress bar
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download and save with progress
        with open(model_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        print(f"\nModel downloaded successfully: {model_path}")
        print(f"File size: {os.path.getsize(model_path)} bytes")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        return False
    except Exception as e:
        print(f"Error saving model: {e}")
        return False
    
    return True

def verify_model_file():
    """Verify the downloaded model file"""
    model_path = "yolov8n.pt"
    
    if not os.path.exists(model_path):
        print("Model file does not exist")
        return False
    
    file_size = os.path.getsize(model_path)
    print(f"Model file size: {file_size} bytes")
    
    # YOLOv8n model should be around 6-7MB
    if file_size < 6000000:  # Less than 6MB
        print("Warning: Model file seems too small, might be corrupted")
        return False
    
    print("Model file appears to be valid")
    return True

if __name__ == "__main__":
    print("YOLOv8 Model Downloader")
    print("=" * 30)
    
    # Check if model already exists
    if os.path.exists("yolov8n.pt"):
        print("Model file already exists. Verifying...")
        if verify_model_file():
            print("Existing model file appears valid.")
            response = input("Do you want to download a fresh copy? (y/n): ")
            if response.lower() != 'y':
                print("Keeping existing model file.")
                exit(0)
        else:
            print("Existing model file appears corrupted. Downloading fresh copy...")
    
    # Download the model
    if download_yolov8_model():
        print("Download completed successfully!")
        print("You can now run: streamlit run room.py")
    else:
        print("Download failed. Please check your internet connection and try again.")
