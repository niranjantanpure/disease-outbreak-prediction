# disease_outbreak_prediction/src/models/outbreak_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class OutbreakPredictor:
    """
    Machine Learning pipeline for predicting disease outbreaks
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.encoders = {}
        self.feature_importance = {}

    def load_and_prepare_data(self, social_data_file, health_data_file):
        """
        Load and combine social media and health data
        """
        # Load social media data
        social_df = pd.read_csv(social_data_file)

        # Load health data
        import json
        with open(health_data_file, 'r') as f:
            health_data = json.load(f)

        # Process surveillance data
        surveillance_df = pd.DataFrame(health_data['surveillance_data'])
        surveillance_df['date'] = pd.to_datetime(surveillance_df[['year', 'week']].apply(
            lambda x: datetime.strptime(f"{int(x['year'])}-W{int(x['week'])}-1", "%Y-W%W-%w"), axis=1))

        # Process environmental data
        env_df = pd.DataFrame(health_data['environmental_data'])
        env_df['date'] = pd.to_datetime(env_df['date'])

        return social_df, surveillance_df, env_df

    def create_outbreak_features(self, social_df, surveillance_df, env_df):
        """
        Create comprehensive feature set for outbreak prediction
        """
        # Group social media data by date
        social_df['created_at'] = pd.to_datetime(social_df['created_at'])
        social_df['date'] = social_df['created_at'].dt.date

        # Social media features (daily aggregation)
        social_features = social_df.groupby('date').agg({
            'polarity': ['mean', 'std', 'min', 'max'],
            'subjectivity': ['mean', 'std'],
            'engagement_score': ['sum', 'mean', 'max'],
            'id': 'count'  # Number of posts
        }).round(4)

        # Flatten column names
        social_features.columns = [
            '_'.join(col).strip() for col in social_features.columns]
        social_features = social_features.reset_index()
        social_features['date'] = pd.to_datetime(social_features['date'])

        # Add symptom mention counts
        symptom_counts = social_df.groupby('date')['symptoms'].apply(
            lambda x: sum([len(eval(symptoms)) if isinstance(
                symptoms, str) else 0 for symptoms in x])
        ).reset_index()
        symptom_counts.columns = ['date', 'total_symptom_mentions']
        symptom_counts['date'] = pd.to_datetime(symptom_counts['date'])

        # Merge social features
        social_features = social_features.merge(
            symptom_counts, on='date', how='left')

        # Prepare surveillance data (weekly to daily interpolation)
        surveillance_daily = surveillance_df.set_index(
            'date').resample('D').interpolate()
        surveillance_daily = surveillance_daily.reset_index()

        # Prepare environmental data
        env_daily = env_df.copy()

        # Merge all datasets
        # Start with surveillance data as base
        combined_df = surveillance_daily.copy()

        # Merge social media features
        combined_df = combined_df.merge(social_features, on='date', how='left')

        # Merge environmental data
        combined_df = combined_df.merge(env_daily, on='date', how='left')

        # Fill missing values
        combined_df = combined_df.fillna(method='forward').fillna(0)

        return combined_df

    def create_outbreak_labels(self, df, outbreak_threshold_percentile=85):
        """
        Create binary outbreak labels based on case counts
        """
        # Calculate outbreak threshold for each disease
        outbreak_labels = []

        for disease in df['disease'].unique():
            disease_data = df[df['disease'] == disease].copy()
            threshold = np.percentile(
                disease_data['cases'], outbreak_threshold_percentile)

            # Create binary labels
            disease_data['outbreak'] = (
                disease_data['cases'] > threshold).astype(int)

            # Add rolling averages for better signal
            disease_data['cases_7day_avg'] = disease_data['cases'].rolling(
                window=7, center=True).mean()
            disease_data['cases_trend'] = disease_data['cases_7day_avg'].diff()

            outbreak_labels.append(disease_data)

        return pd.concat(outbreak_labels, ignore_index=True)

    def engineer_features(self, df):
        """
        Create additional engineered features
        """
        # Sort by disease and date
        df = df.sort_values(['disease', 'date'])

        # Time-based features
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # Seasonal features
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3  # Fall
        })

        # Disease-specific lagged features
        for disease in df['disease'].unique():
            mask = df['disease'] == disease

            # Lagged case counts
            df.loc[mask, 'cases_lag_1'] = df.loc[mask, 'cases'].shift(1)
            df.loc[mask, 'cases_lag_7'] = df.loc[mask, 'cases'].shift(7)
            df.loc[mask, 'cases_lag_14'] = df.loc[mask, 'cases'].shift(14)

            # Rolling statistics
            df.loc[mask, 'cases_mean_7d'] = df.loc[mask,
                                                   'cases'].rolling(7).mean()
            df.loc[mask, 'cases_std_7d'] = df.loc[mask,
                                                  'cases'].rolling(7).std()
            df.loc[mask, 'cases_mean_30d'] = df.loc[mask,
                                                    'cases'].rolling(30).mean()

            # Social media lagged features
            if 'polarity_mean' in df.columns:
                df.loc[mask, 'polarity_lag_1'] = df.loc[mask,
                                                        'polarity_mean'].shift(1)
                df.loc[mask, 'polarity_lag_3'] = df.loc[mask,
                                                        'polarity_mean'].shift(3)
                df.loc[mask, 'symptom_mentions_lag_1'] = df.loc[mask,
                                                                'total_symptom_mentions'].shift(1)

        # Interaction features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

        if 'polarity_mean' in df.columns and 'total_symptom_mentions' in df.columns:
            df['sentiment_symptoms_interaction'] = df['polarity_mean'] * \
                df['total_symptom_mentions']

        return df

    def prepare_features_and_target(self, df):
        """
        Prepare final feature matrix and target variable
        """
        # Select feature columns
        exclude_cols = ['date', 'outbreak', 'disease',
                        'cases', 'deaths', 'hospitalizations']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Handle categorical variables
        categorical_cols = df[feature_cols].select_dtypes(
            include=['object']).columns

        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.encoders[col].transform(df[col].astype(str))

        # Prepare feature matrix
        X = df[feature_cols].fillna(0)
        y = df['outbreak']

        return X, y, feature_cols

    def train_models(self, X, y, test_size=0.2, random_state=42):
        """
        Train multiple ML models for outbreak prediction
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)

        # Feature selection
        self.feature_selectors['kbest'] = SelectKBest(
            score_func=f_classif, k=min(20, X_train.shape[1]))
        X_train_selected = self.feature_selectors['kbest'].fit_transform(
            X_train_scaled, y_train)
        X_test_selected = self.feature_selectors['kbest'].transform(
            X_test_scaled)

        # Define models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=random_state),
            'logistic_regression': LogisticRegression(random_state=random_state),
            'svm': SVC(probability=True, random_state=random_state)
        }

        # Train and evaluate models
        results = {}

        for name, model in models.items():
            print(f"Training {name}...")

            # Use selected features for linear models, all features for tree-based
            if name in ['logistic_regression', 'svm']:
                X_train_model = X_train_selected
                X_test_model = X_test_selected
            else:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled

            # Train model
            model.fit(X_train_model, y_train)

            # Predict
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]

            # Evaluate
            cv_scores = cross_val_score(
                model, X_train_model, y_train, cv=5, scoring='roc_auc')

            results[name] = {
                'model': model,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'test_accuracy': (y_pred == y_test).mean(),
                'test_auc': roc_auc_score(y_test, y_pred_proba),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            # Store model
            self.models[name] = model

            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                if name in ['logistic_regression', 'svm']:
                    feature_names = [f'feature_{i}' for i in range(
                        X_train_selected.shape[1])]
                else:
                    feature_names = X.columns

                self.feature_importance[name] = dict(
                    zip(feature_names, model.feature_importances_))

        # Store test data for later analysis
        self.test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'results': results
        }

        return results

    def hyperparameter_tuning(self, X, y, model_name='random_forest'):
        """
        Perform hyperparameter tuning for selected model
        """
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)

        elif model_name == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42)

        else:
            raise ValueError(
                f"Hyperparameter tuning not implemented for {model_name}")

        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )

        # Scale features
        X_scaled = self.scalers['standard'].transform(X)
        grid_search.fit(X_scaled, y)

        # Store best model
        self.models[f'{model_name}_tuned'] = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.best_score_

    def predict_outbreak(self, X, model_name='random_forest'):
        """
        Predict outbreak probability for new data
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        # Prepare features
        X_scaled = self.scalers['standard'].transform(X)

        # Select features if needed
        if model_name in ['logistic_regression', 'svm']:
            X_scaled = self.feature_selectors['kbest'].transform(X_scaled)

        # Predict
        predictions = self.models[model_name].predict(X_scaled)
        probabilities = self.models[model_name].predict_proba(X_scaled)[:, 1]

        return predictions, probabilities

    def evaluate_models(self):
        """
        Generate comprehensive evaluation report
        """
        if not hasattr(self, 'test_data'):
            raise ValueError("No test data available. Train models first.")

        results = self.test_data['results']
        y_test = self.test_data['y_test']

        # Print performance summary
        print("Model Performance Summary:")
        print("-" * 60)

        for name, result in results.items():
            print(f"{name:20} | CV AUC: {result['cv_score_mean']:.3f} ± {result['cv_score_std']:.3f} | "
                  f"Test AUC: {result['test_auc']:.3f} | Test Acc: {result['test_accuracy']:.3f}")

        # Generate detailed reports
        for name, result in results.items():
            print(f"\n{name.upper()} Detailed Report:")
            print(classification_report(y_test, result['predictions']))

    def plot_feature_importance(self, model_name='random_forest', top_k=15):
        """
        Plot feature importance for tree-based models
        """
        if model_name not in self.feature_importance:
            print(f"No feature importance available for {model_name}")
            return

        importance = self.feature_importance[model_name]

        # Sort features by importance
        sorted_features = sorted(
            importance.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Create plot
        features, importances = zip(*sorted_features)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def save_models(self, filepath='outbreak_models.joblib'):
        """
        Save trained models and preprocessors
        """
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance
        }

        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")

    def load_models(self, filepath='outbreak_models.joblib'):
        """
        Load trained models and preprocessors
        """
        model_data = joblib.load(filepath)

        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_selectors = model_data['feature_selectors']
        self.encoders = model_data['encoders']
        self.feature_importance = model_data['feature_importance']

        print(f"Models loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = OutbreakPredictor()

    # For demonstration, create synthetic data
    # In real implementation, use your collected data files

    print("Disease Outbreak Prediction Pipeline Demo")
    print("This is a template - replace with your actual data files")
