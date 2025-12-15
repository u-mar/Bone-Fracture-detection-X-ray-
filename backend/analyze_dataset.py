import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os

def analyze_and_sample_dataset():
    # Read the original train.csv
    print("Reading train.csv...")
    df = pd.read_csv('C:/Users/M salah/Downloads/train.csv')
    
    # Print original distribution
    print("\nOriginal Dataset Distribution:")
    print("-" * 50)
    print(df['body_part'].value_counts())
    
    # Create a balanced sample
    sample_size = 100  # Number of images per category
    
    # Get a balanced sample
    sampled_df = df.groupby('body_part').apply(
        lambda x: x.sample(min(len(x), sample_size), random_state=42)
    ).reset_index(drop=True)
    
    # Print sampled distribution
    print("\nSampled Dataset Distribution:")
    print("-" * 50)
    print(sampled_df['body_part'].value_counts())
    
    # Save the sampled dataset
    output_path = 'train_sample.csv'
    sampled_df.to_csv(output_path, index=False)
    print(f"\nSaved balanced sample to {output_path}")
    
    # Create directories for the dataset
    base_dir = Path('dataset/bodyparts')
    
    # Map body parts to our categories
    body_part_mapping = {
        'HAND': 'hand',
        'FOREARM': 'arm',
        'ARM': 'arm',
        'ELBOW': 'arm',
        'SHOULDER': 'arm',
        'FOOT': 'foot',
        'LEG': 'leg',
        'KNEE': 'leg',
        'THIGH': 'leg',
        'CHEST': 'chest',
        'SPINE': 'spine',
        'SKULL': 'skull'
    }
    
    # Print image IDs for each category
    print("\nImage IDs for each category:")
    print("-" * 50)
    for body_part in sampled_df['body_part'].unique():
        print(f"\n{body_part}:")
        image_ids = sampled_df[sampled_df['body_part'] == body_part]['image_id'].tolist()
        print(f"Number of images: {len(image_ids)}")
        print("First 5 image IDs:", image_ids[:5])
    
    print("\nNext steps:")
    print("1. Use these image IDs to download the specific images from Kaggle")
    print("2. The sample is balanced with approximately 100 images per category")
    print("3. Place the downloaded images in 'dataset/bodyparts/train/{category}' folders")
    print(f"4. You can find the list of image IDs in {output_path}")

if __name__ == "__main__":
    analyze_and_sample_dataset()