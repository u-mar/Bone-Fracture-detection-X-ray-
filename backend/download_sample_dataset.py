import os
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

# Create necessary directories
downloads_dir = Path('downloads')
downloads_dir.mkdir(exist_ok=True)

# URL for train.csv from the competition
TRAIN_CSV_URL = "https://raw.githubusercontent.com/Unifesp/xray-body-part-detection/main/train.csv"

def download_train_csv():
    """Download the train.csv file"""
    print("Downloading train.csv...")
    try:
        response = requests.get(TRAIN_CSV_URL)
        if response.status_code == 200:
            with open(downloads_dir / 'train.csv', 'wb') as f:
                f.write(response.content)
            print("Successfully downloaded train.csv")
            return True
        else:
            print("Failed to download train.csv")
            return False
    except Exception as e:
        print(f"Error downloading train.csv: {str(e)}")
        return False

def analyze_dataset():
    """Analyze the dataset distribution"""
    try:
        df = pd.read_csv(downloads_dir / 'train.csv')
        print("\nDataset Statistics:")
        print("-" * 50)
        print("\nBody Part Distribution:")
        print(df['body_part'].value_counts())
        
        # Save a smaller sample
        sample_size = 1000  # Adjust this number based on how many images you want
        stratified_sample = df.groupby('body_part', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(sample_size/len(df['body_part'].unique()))))
        )
        
        print(f"\nCreating a balanced sample of ~{sample_size} images...")
        stratified_sample.to_csv(downloads_dir / 'train_sample.csv', index=False)
        print("\nSample dataset statistics:")
        print(stratified_sample['body_part'].value_counts())
        
        print("\nNext steps:")
        print("1. Go to the Kaggle competition page")
        print("2. Download only the images listed in train_sample.csv")
        print(f"3. Place the downloaded images in {downloads_dir/'train'} directory")
        print("4. Run the training script")
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")

def main():
    if download_train_csv():
        analyze_dataset()
    
    print("\nImportant Note:")
    print("To download specific images from the dataset:")
    print("1. Use train_sample.csv which contains a balanced subset of images")
    print("2. Only download the images listed in this file")
    print("3. This will significantly reduce the download size")

if __name__ == "__main__":
    main()