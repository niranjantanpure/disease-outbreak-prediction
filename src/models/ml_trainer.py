#!/usr/bin/env python3
"""
Machine Learning model training module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import xgboost as xgb
import joblib
from datetime import datetime


class MLTrainer:
    """Train and evaluate machine learning models for outbreak prediction"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.X_test = None
        self.y_test = None
        self.feature_names = None

    def setup_models(self):
        """Initialize ML models"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'SVM': SVC(
                probability=True,
                random_state=self.random_state
            )
        }

        print(f"✓ Initialized {len(self.models)} ML models")

    def split_and_scale_data(self, X, y, test_size=0.2):
        """Split data into train/test and scale features"""
        print("🔄 Splitting and scaling data...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Store for later use
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = X.columns

        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        print(
            f"✓ Target distribution - Train: {y_train.value_counts().to_dict()}")
        print(
            f"✓ Target distribution - Test: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

    def train_single_model(self, name, model, X_train_orig, X_train_scaled, X_test_orig, X_test_scaled, y_train, y_test, use_scaled=False):
        """Train a single model and return results"""
        print(f"  🔄 Training {name}...")

        # Choose scaled or original features
        if use_scaled:
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train_orig
            X_test_model = X_test_orig

        # Train model
        model.fit(X_train_model, y_train)

        # Predictions
        y_pred = model.predict(X_test_model)
        y_pred_proba = model.predict_proba(X_test_model)[:, 1]

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_model, y_train, cv=5, scoring='roc_auc'
        )

        # Metrics
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_accuracy = (y_pred == y_test).mean()

        result = {
            'model': model,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'test_auc': test_auc,
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'cv_scores': cv_scores
        }

        print(f"    ✓ CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"    ✓ Test AUC: {test_auc:.3f}")
        print(f"    ✓ Accuracy: {test_accuracy:.3f}")

        return result

    def train_all_models(self, X, y):
        """Train all models and return results"""
        print("🤖 Training all ML models...")

        if not self.models:
            self.setup_models()

        # Split and scale data
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = self.split_and_scale_data(
            X, y)

        # Train each model
        for name, model in self.models.items():
            # Use scaled features for linear models
            use_scaled = name in ['Logistic Regression', 'SVM']

            result = self.train_single_model(
                name, model, X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test, use_scaled
            )

            self.results[name] = result

        print("✅ All models trained successfully!")
        return self.results

    def get_performance_summary(self):
        """Get performance summary of all models"""
        if not self.results:
            raise ValueError(
                "No models trained yet. Call train_all_models() first.")

        performance_data = []
        for name, result in self.results.items():
            performance_data.append({
                'Model': name,
                'CV_AUC': result['cv_auc_mean'],
                'CV_Std': result['cv_auc_std'],
                'Test_AUC': result['test_auc'],
                'Test_Accuracy': result['test_accuracy']
            })

        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.sort_values(
            'Test_AUC', ascending=False)

        return performance_df

    def get_best_model(self):
        """Get the best performing model"""
        performance_df = self.get_performance_summary()
        best_model_name = performance_df.iloc[0]['Model']
        best_result = self.results[best_model_name]

        return best_model_name, best_result

    def get_feature_importance(self, model_name=None):
        """Get feature importance for tree-based models"""
        if model_name is None:
            model_name, _ = self.get_best_model()

        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")

        model = self.results[model_name]['model']

        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not have feature importance")
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def get_classification_report(self, model_name=None):
        """Get detailed classification report"""
        if model_name is None:
            model_name, _ = self.get_best_model()

        result = self.results[model_name]

        report = classification_report(
            self.y_test, result['predictions'],
            target_names=['Normal', 'Outbreak'],
            output_dict=True
        )

        return report

    def get_confusion_matrix(self, model_name=None):
        """Get confusion matrix"""
        if model_name is None:
            model_name, _ = self.get_best_model()

        result = self.results[model_name]
        cm = confusion_matrix(self.y_test, result['predictions'])

        return cm

    def hyperparameter_tuning(self, model_name='Random Forest', param_grid=None):
        """Perform hyperparameter tuning for a specific model"""
        print(f"🔧 Hyperparameter tuning for {model_name}...")

        if param_grid is None:
            if model_name == 'Random Forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_name == 'XGBoost':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            else:
                raise ValueError(f"No default param_grid for {model_name}")

        # Get base model
        base_model = self.models[model_name]

        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )

        # Use appropriate data (scaled for linear models)
        if model_name in ['Logistic Regression', 'SVM']:
            X_train_model = self.scaler.fit_transform(
                self.X_test)  # This should be proper training data
        else:
            X_train_model = self.X_test  # This should be proper training data

        grid_search.fit(X_train_model, self.y_test)

        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score: {grid_search.best_score_:.3f}")

        return grid_search.best_params_, grid_search.best_score_

    def save_models(self, filepath='models/trained_models.joblib'):
        """Save all trained models"""
        model_data = {
            'models': {name: result['model'] for name, result in self.results.items()},
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'results': self.results,
            'training_date': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)
        print(f"✓ Models saved to {filepath}")

    def load_models(self, filepath='models/trained_models.joblib'):
        """Load trained models"""
        model_data = joblib.load(filepath)

        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.results = model_data['results']

        print(f"✓ Models loaded from {filepath}")
        print(f"✓ Training date: {model_data['training_date']}")


if __name__ == "__main__":
    # Test the ML trainer
    import sys
    import os
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), '..', 'data_processing'))

    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer

    # Load and prepare data
    loader = DataLoader()
    social_df, surveillance_df, env_df = loader.load_all_data()

    engineer = FeatureEngineer()
    X, y, feature_cols, ml_data = engineer.prepare_ml_features(
        surveillance_df, env_df, social_df
    )

    # Train models
    trainer = MLTrainer()
    results = trainer.train_all_models(X, y)

    # Show results
    performance_df = trainer.get_performance_summary()
    print("\n📊 Model Performance Summary:")
    print(performance_df.round(3))

    best_model_name, best_result = trainer.get_best_model()
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📈 Test AUC: {best_result['test_auc']:.3f}")
