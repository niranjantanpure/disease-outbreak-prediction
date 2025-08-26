#!/usr/bin/env python3
"""
Feature engineering module for outbreak prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


class FeatureEngineer:
    """Create features for machine learning from raw data"""

    def __init__(self):
        self.encoders = {}

    def aggregate_social_media(self, social_df):
        """Aggregate social media data by date"""
        print("🔄 Aggregating social media data...")

        # Create date column
        social_df['date'] = social_df['created_at'].dt.date

        # Aggregate by date
        social_daily = social_df.groupby('date').agg({
            'polarity': ['mean', 'std', 'min', 'max'],
            'subjectivity': ['mean', 'std'],
            'engagement_score': ['sum', 'mean', 'max'],
            'id': 'count'  # Number of posts
        }).round(4)

        # Flatten column names
        social_daily.columns = ['_'.join(col).strip()
                                for col in social_daily.columns]
        social_daily = social_daily.reset_index()
        social_daily['date'] = pd.to_datetime(social_daily['date'])

        # Add symptom counts
        social_daily['symptom_mentions'] = social_daily['date'].apply(
            lambda x: self._count_symptoms_for_date(social_df, x.date())
        )

        print(f"✓ Created {len(social_daily)} daily social media aggregations")
        return social_daily

    def _count_symptoms_for_date(self, social_df, date_val):
        """Count symptoms mentioned on a specific date"""
        date_posts = social_df[social_df['created_at'].dt.date == date_val]
        total_symptoms = 0

        for symptoms_str in date_posts['symptoms']:
            try:
                symptoms = eval(symptoms_str)
                total_symptoms += len(symptoms)
            except:
                pass

        return total_symptoms

    def add_time_features(self, df, date_col='date'):
        """Add time-based features"""
        print("📅 Adding time-based features...")

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Time features
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['day_of_year'] = df[date_col].dt.dayofyear

        # Seasonal features
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3  # Fall
        })

        print("✓ Added time-based features")
        return df

    def add_lagged_features(self, df, group_col='disease'):
        """Add lagged features for time series"""
        print("⏳ Adding lagged features...")

        df = df.copy()
        df = df.sort_values([group_col, 'date'])

        for group in df[group_col].unique():
            mask = df[group_col] == group

            # Lagged case counts
            df.loc[mask, 'cases_lag_1'] = df.loc[mask, 'cases'].shift(1)
            df.loc[mask, 'cases_lag_2'] = df.loc[mask, 'cases'].shift(2)

            # Rolling statistics
            df.loc[mask, 'cases_mean_4w'] = df.loc[mask,
                                                   'cases'].rolling(4).mean()
            df.loc[mask, 'cases_std_4w'] = df.loc[mask,
                                                  'cases'].rolling(4).std()
            df.loc[mask, 'cases_trend'] = df.loc[mask, 'cases'].diff()

            # Social media lagged features (if available)
            if 'polarity_mean' in df.columns:
                df.loc[mask, 'polarity_lag_1'] = df.loc[mask,
                                                        'polarity_mean'].shift(1)
                df.loc[mask, 'sentiment_trend'] = df.loc[mask,
                                                         'polarity_mean'].diff()

        print("✓ Added lagged features")
        return df

    def create_interaction_features(self, df):
        """Create interaction features"""
        print("🔗 Creating interaction features...")

        df = df.copy()

        # Environmental interactions
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            df['temp_squared'] = df['temperature'] ** 2

        # Social media interactions
        if 'polarity_mean' in df.columns and 'symptom_mentions' in df.columns:
            df['sentiment_symptoms_interaction'] = df['polarity_mean'] * \
                df['symptom_mentions']

        if 'engagement_score_sum' in df.columns and 'id_count' in df.columns:
            df['avg_engagement_per_post'] = df['engagement_score_sum'] / \
                (df['id_count'] + 1)

        print("✓ Created interaction features")
        return df

    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        print("🏷️ Encoding categorical features...")

        df = df.copy()

        # Encode disease if present
        if 'disease' in df.columns:
            if 'disease' not in self.encoders:
                self.encoders['disease'] = LabelEncoder()
                df['disease_encoded'] = self.encoders['disease'].fit_transform(
                    df['disease'])
            else:
                df['disease_encoded'] = self.encoders['disease'].transform(
                    df['disease'])

        # Handle any other object columns
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['date', 'disease']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        print("✓ Encoded categorical features")
        return df

    def merge_all_datasets(self, surveillance_df, env_df, social_daily):
        """Merge all datasets for ML"""
        print("🔗 Merging all datasets...")

        # Start with surveillance data
        ml_data = surveillance_df.copy()

        # Merge environmental data
        ml_data = ml_data.merge(env_df, on='date', how='left')

        # Merge social media data
        ml_data = ml_data.merge(social_daily, on='date', how='left')

        # Fill missing values
        ml_data = ml_data.ffill().fillna(0)

        print(f"✓ Combined dataset shape: {ml_data.shape}")
        return ml_data

    def prepare_ml_features(self, surveillance_df, env_df, social_df):
        """Complete feature engineering pipeline"""
        print("🔧 Running complete feature engineering pipeline...")

        # Step 1: Aggregate social media
        social_daily = self.aggregate_social_media(social_df)

        # Step 2: Add time features to surveillance data
        surveillance_featured = self.add_time_features(surveillance_df)

        # Step 3: Merge all datasets
        ml_data = self.merge_all_datasets(
            surveillance_featured, env_df, social_daily)

        # Step 4: Add lagged features
        ml_data = self.add_lagged_features(ml_data)

        # Step 5: Create interaction features
        ml_data = self.create_interaction_features(ml_data)

        # Step 6: Encode categorical features
        ml_data = self.encode_categorical_features(ml_data)

        # Step 7: Select features for ML
        exclude_cols = [
            'date', 'disease', 'outbreak', 'cases', 'deaths', 'hospitalizations',
            'outbreak_threshold', 'data_quality', 'source_x', 'source_y', 'region'
        ]
        feature_cols = [
            col for col in ml_data.columns if col not in exclude_cols]

        X = ml_data[feature_cols].fillna(0)
        y = ml_data['outbreak']

        print(f"✅ Feature engineering completed!")
        print(f"📊 Final dataset shape: {X.shape}")
        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")

        return X, y, feature_cols, ml_data


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import DataLoader

    loader = DataLoader()
    social_df, surveillance_df, env_df = loader.load_all_data()

    engineer = FeatureEngineer()
    X, y, feature_cols, ml_data = engineer.prepare_ml_features(
        surveillance_df, env_df, social_df
    )

    print(f"\n📋 Features created: {len(feature_cols)}")
    print(f"🎯 Outbreak rate: {y.mean():.1%}")
