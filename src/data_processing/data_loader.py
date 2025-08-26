#!/usr/bin/env python3
"""
Data loading and preprocessing module
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handle all data loading and basic preprocessing"""

    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
        self.social_df = None
        self.surveillance_df = None
        self.env_df = None
        self.ml_data = None

    def load_social_media_data(self):
        """Load and preprocess social media data"""
        print("📱 Loading social media data...")

        self.social_df = pd.read_csv(f'{self.data_dir}/social_media_data.csv')
        self.social_df['created_at'] = pd.to_datetime(
            self.social_df['created_at'])

        print(f"✓ Loaded {len(self.social_df)} social media records")
        return self.social_df

    def load_surveillance_data(self):
        """Load health surveillance data"""
        print("🏥 Loading surveillance data...")

        self.surveillance_df = pd.read_csv(
            f'{self.data_dir}/surveillance_with_labels.csv')
        self.surveillance_df['date'] = pd.to_datetime(
            self.surveillance_df['date'])

        print(f"✓ Loaded {len(self.surveillance_df)} surveillance records")
        return self.surveillance_df

    def load_environmental_data(self):
        """Load environmental data"""
        print("🌡️ Loading environmental data...")

        with open(f'{self.data_dir}/health_data.json', 'r') as f:
            health_data = json.load(f)

        self.env_df = pd.DataFrame(health_data['environmental_data'])
        self.env_df['date'] = pd.to_datetime(self.env_df['date'])

        print(f"✓ Loaded {len(self.env_df)} environmental records")
        return self.env_df

    def load_all_data(self):
        """Load all datasets"""
        print("📊 Loading all datasets...")

        self.load_social_media_data()
        self.load_surveillance_data()
        self.load_environmental_data()

        print("✅ All datasets loaded successfully!")
        return self.social_df, self.surveillance_df, self.env_df

    def get_data_summary(self):
        """Get summary of all loaded datasets"""
        if any(df is None for df in [self.social_df, self.surveillance_df, self.env_df]):
            self.load_all_data()

        summary = {
            'social_media': {
                'records': len(self.social_df),
                'date_range': (self.social_df['created_at'].min(), self.social_df['created_at'].max()),
                'platforms': self.social_df['platform'].unique().tolist(),
                'avg_sentiment': self.social_df['polarity'].mean()
            },
            'surveillance': {
                'records': len(self.surveillance_df),
                'date_range': (self.surveillance_df['date'].min(), self.surveillance_df['date'].max()),
                'diseases': self.surveillance_df['disease'].unique().tolist(),
                'outbreak_rate': self.surveillance_df['outbreak'].mean()
            },
            'environmental': {
                'records': len(self.env_df),
                'date_range': (self.env_df['date'].min(), self.env_df['date'].max()),
                'temp_range': (self.env_df['temperature'].min(), self.env_df['temperature'].max()),
                'avg_humidity': self.env_df['humidity'].mean()
            }
        }

        return summary


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    social_df, surveillance_df, env_df = loader.load_all_data()

    summary = loader.get_data_summary()
    print("\n📋 Data Summary:")
    for dataset, info in summary.items():
        print(f"\n{dataset.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
