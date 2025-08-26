#!/usr/bin/env python3
"""
Visualization module for outbreak prediction analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter


class OutbreakVisualizer:
    """Create visualizations for outbreak prediction analysis"""

    def __init__(self, style='seaborn-v0_8'):
        plt.style.use(style)
        sns.set_palette('husl')

    def plot_data_overview(self, social_df, surveillance_df, env_df):
        """Create overview plots of all datasets"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Social media sentiment
        axes[0, 0].hist(social_df['polarity'], bins=20,
                        alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Social Media Sentiment Distribution')
        axes[0, 0].set_xlabel('Polarity (Negative ← → Positive)')
        axes[0, 0].set_ylabel('Frequency')

        # Platform distribution
        platform_counts = social_df['platform'].value_counts()
        axes[0, 1].pie(platform_counts.values,
                       labels=platform_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Posts by Platform')

        # Cases by disease
        disease_means = surveillance_df.groupby('disease')['cases'].mean()
        bars = axes[0, 2].bar(disease_means.index,
                              disease_means.values, color='lightgreen')
        axes[0, 2].set_title('Average Cases by Disease')
        axes[0, 2].set_ylabel('Average Weekly Cases')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Temperature over time
        env_df_sample = env_df.iloc[::7]  # Sample every 7th day for clarity
        axes[1, 0].plot(pd.to_datetime(env_df_sample['date']), env_df_sample['temperature'],
                        color='red', alpha=0.7)
        axes[1, 0].set_title('Temperature Over Time')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Outbreak frequency
        outbreak_counts = surveillance_df.groupby('disease')['outbreak'].sum()
        axes[1, 1].bar(outbreak_counts.index,
                       outbreak_counts.values, color='orange')
        axes[1, 1].set_title('Outbreak Weeks by Disease')
        axes[1, 1].set_ylabel('Number of Outbreak Weeks')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Environmental correlation
        env_corr = env_df[['temperature', 'humidity',
                           'precipitation', 'air_quality_index']].corr()
        sns.heatmap(env_corr, annot=True, cmap='coolwarm',
                    center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Environmental Variables Correlation')

        plt.tight_layout()
        plt.show()

    def plot_symptom_analysis(self, social_df):
        """Analyze and plot symptom mentions"""
        # Extract all symptoms
        all_symptoms = []
        for symptoms_str in social_df['symptoms']:
            try:
                symptoms = eval(symptoms_str)
                all_symptoms.extend(symptoms)
            except:
                pass

        symptom_counts = Counter(all_symptoms)
        top_symptoms = symptom_counts.most_common(10)

        if not top_symptoms:
            print("No symptoms found in the data")
            return

        symptoms, counts = zip(*top_symptoms)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(symptoms, counts, color='lightcoral')
        plt.title('Most Mentioned Health Symptoms in Social Media')
        plt.xlabel('Symptoms')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)

        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        return symptom_counts

    def plot_time_series_analysis(self, surveillance_df):
        """Plot time series of disease cases"""
        plt.figure(figsize=(15, 8))

        for disease in surveillance_df['disease'].unique():
            disease_data = surveillance_df[surveillance_df['disease'] == disease]
            plt.plot(disease_data['date'], disease_data['cases'],
                     label=disease, alpha=0.8, linewidth=2)

        plt.title('Disease Cases Over Time')
        plt.xlabel('Date')
        plt.ylabel('Weekly Cases')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_model_performance(self, results):
        """Plot model performance comparison"""
        # Prepare data
        models = list(results.keys())
        cv_aucs = [results[model]['cv_auc_mean'] for model in models]
        test_aucs = [results[model]['test_auc'] for model in models]
        accuracies = [results[model]['test_accuracy'] for model in models]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # AUC comparison
        x = np.arange(len(models))
        width = 0.35

        bars1 = axes[0].bar(x - width/2, cv_aucs, width,
                            label='CV AUC', alpha=0.8)
        bars2 = axes[0].bar(x + width/2, test_aucs, width,
                            label='Test AUC', alpha=0.8)

        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('AUC Score')
        axes[0].set_title('Model AUC Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # Accuracy comparison
        bars = axes[1].bar(models, accuracies, color='lightgreen', alpha=0.8)
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        for bar, acc in zip(bars, accuracies):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                         f'{acc:.3f}', ha='center', va='bottom')

        # Performance summary
        performance_data = pd.DataFrame({
            'Model': models,
            'CV_AUC': cv_aucs,
            'Test_AUC': test_aucs,
            'Accuracy': accuracies
        }).sort_values('Test_AUC', ascending=False)

        # Create table
        axes[2].axis('tight')
        axes[2].axis('off')
        table = axes[2].table(cellText=performance_data.round(3).values,
                              colLabels=performance_data.columns,
                              cellLoc='center',
                              loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[2].set_title('Performance Summary')

        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self, results, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))

        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            plt.plot(
                fpr, tpr, label=f"{name} (AUC = {result['test_auc']:.3f})", linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Outbreak'],
                    yticklabels=['Normal', 'Outbreak'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def plot_feature_importance(self, importance_df, model_name, top_k=15):
        """Plot feature importance"""
        if importance_df is None:
            print(f"No feature importance available for {model_name}")
            return

        top_features = importance_df.head(top_k)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_outbreak_patterns(self, ml_data):
        """Analyze outbreak patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Outbreak by season
        outbreak_by_season = ml_data.groupby('season')['outbreak'].mean()
        season_names = ['Winter', 'Spring', 'Summer', 'Fall']

        axes[0, 0].bar(season_names, outbreak_by_season.values,
                       color='lightblue')
        axes[0, 0].set_title('Outbreak Rate by Season')
        axes[0, 0].set_ylabel('Outbreak Rate')

        # Cases distribution by outbreak status
        outbreak_cases = ml_data[ml_data['outbreak'] == 1]['cases']
        normal_cases = ml_data[ml_data['outbreak'] == 0]['cases']

        axes[0, 1].hist([normal_cases, outbreak_cases], bins=20, alpha=0.7,
                        label=['Normal', 'Outbreak'], color=['lightgreen', 'orange'])
        axes[0, 1].set_title('Cases Distribution by Outbreak Status')
        axes[0, 1].set_xlabel('Weekly Cases')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # Temperature vs outbreak
        temp_bins = pd.cut(ml_data['temperature'], bins=5)
        outbreak_by_temp = ml_data.groupby(temp_bins)['outbreak'].mean()

        axes[1, 0].bar(range(len(outbreak_by_temp)),
                       outbreak_by_temp.values, color='red', alpha=0.7)
        axes[1, 0].set_title('Outbreak Rate by Temperature Range')
        axes[1, 0].set_xlabel('Temperature Bins')
        axes[1, 0].set_ylabel('Outbreak Rate')

        # Humidity vs outbreak
        humidity_bins = pd.cut(ml_data['humidity'], bins=5)
        outbreak_by_humidity = ml_data.groupby(
            humidity_bins)['outbreak'].mean()

        axes[1, 1].bar(range(len(outbreak_by_humidity)),
                       outbreak_by_humidity.values, color='blue', alpha=0.7)
        axes[1, 1].set_title('Outbreak Rate by Humidity Range')
        axes[1, 1].set_xlabel('Humidity Bins')
        axes[1, 1].set_ylabel('Outbreak Rate')

        plt.tight_layout()
        plt.show()

    def create_interactive_dashboard(self, ml_data, results):
        """Create interactive Plotly dashboard"""
        # Time series of cases with outbreak indicators
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Disease Cases Over Time', 'Environmental Factors',
                            'Model Performance', 'Outbreak Patterns'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        # Disease cases timeline
        for disease in ml_data['disease'].unique():
            disease_data = ml_data[ml_data['disease'] == disease]

            fig.add_trace(
                go.Scatter(x=disease_data['date'], y=disease_data['cases'],
                           name=f'{disease} Cases', mode='lines'),
                row=1, col=1
            )

        # Environmental factors
        fig.add_trace(
            go.Scatter(x=ml_data['date'], y=ml_data['temperature'],
                       name='Temperature', line=dict(color='red')),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=ml_data['date'], y=ml_data['humidity'],
                       name='Humidity', yaxis='y2', line=dict(color='blue')),
            row=1, col=2, secondary_y=True
        )

        # Model performance
        models = list(results.keys())
        aucs = [results[model]['test_auc'] for model in models]

        fig.add_trace(
            go.Bar(x=models, y=aucs, name='Test AUC'),
            row=2, col=1
        )

        # Outbreak patterns
        fig.add_trace(
            go.Scatter(x=ml_data['temperature'], y=ml_data['humidity'],
                       mode='markers',
                       marker=dict(color=ml_data['outbreak'],
                                   colorscale='RdYlBu', size=8),
                       name='Outbreak Status'),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True,
                          title_text="Disease Outbreak Analysis Dashboard")

        fig.show()


if __name__ == "__main__":
    # Test the visualizer
    import sys
    import os
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), '..', 'data_processing'))

    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer

    # Load data
    loader = DataLoader()
    social_df, surveillance_df, env_df = loader.load_all_data()

    # Create visualizations
    viz = OutbreakVisualizer()

    print("📊 Creating data overview plots...")
    viz.plot_data_overview(social_df, surveillance_df, env_df)

    print("🤒 Analyzing symptoms...")
    viz.plot_symptom_analysis(social_df)

    print("📈 Creating time series analysis...")
    viz.plot_time_series_analysis(surveillance_df)
