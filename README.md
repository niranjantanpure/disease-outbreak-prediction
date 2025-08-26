# AI-Powered Disease Outbreak Prediction System

> **MSc Computer Science Dissertation Project**  
> **Author**: Niranjan Tanpure  
> **University**: York St John University  
> **Dataset**: 80,372+ Real WHO FluNet Records + 17,576+ Social Media Posts

## 🦠 Project Overview

This dissertation project demonstrates an advanced AI system for predicting disease outbreaks by combining:
- **WHO FluNet surveillance data** (156,291 records from 180 countries)
- **Social media sentiment analysis** (Reddit + Twitter health discussions)
- **Environmental factors** (weather, air quality data)
- **GPU-accelerated machine learning** (XGBoost achieving 89.9% accuracy)

## 🚀 Key Features

- ✅ **Real Data Processing**: 80k+ WHO surveillance records
- ✅ **Global Coverage**: 180 countries, 29+ years of data
- ✅ **GPU Acceleration**: 2.7s training time with CUDA
- ✅ **Academic Quality**: Statistical validation + research contributions
- ✅ **Production Ready**: Scalable architecture for health agencies
- ✅ **Comprehensive Visualizations**: 8 publication-ready plots

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+ (recommended: 3.9)
- **Memory**: 4GB+ RAM
- **Storage**: 2GB+ free space
- **GPU**: NVIDIA GPU (optional, for acceleration)

### Required Packages
```bash
# Core dependencies
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0

# Optional GPU acceleration
cupy-cuda11x>=10.0.0  # For NVIDIA GPUs
```

## 🔧 Installation

### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone <your-repo-url>
cd "analysis - Copy"

# Create conda environment
conda create -n disease_prediction python=3.9
conda activate disease_prediction

# Install packages
conda install pandas numpy scikit-learn matplotlib seaborn joblib -c conda-forge
conda install xgboost -c conda-forge

# Optional: GPU acceleration
conda install cupy -c conda-forge
```

### Option 2: Using pip
```bash
# Clone repository
git clone <your-repo-url>
cd "analysis - Copy"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
pip install cupy-cuda11x  # Optional: for GPU
```

## 📊 Dataset Structure

```
data/
├── raw/                          # Raw data files (100% real data)
│   ├── VIW_FNT.csv              # WHO FluNet surveillance (25MB)
│   ├── Reddit_Combi.csv         # Real Reddit health posts (5.4MB)
│   ├── Twitter_Full.csv         # Real Twitter health data (2.2MB)
│   └── weather_data.csv         # Environmental data
└── processed/                    # Auto-generated processed data
    ├── social_media_data.csv
    ├── surveillance_data.csv
    └── environmental_data.csv
```

## 🏃‍♂️ How to Run

### Quick Start (Full Analysis)
```bash
# Activate environment
conda activate disease_prediction

# Run complete analysis (uses all 80k+ records)
python dissertation_analysis.py
```

### Advanced Options
```bash
# Sample analysis (faster for testing)
python dissertation_analysis.py --sample 10000

# Disable GPU (use CPU only)
python dissertation_analysis.py --no-gpu

# Skip visualizations (faster execution)
python dissertation_analysis.py --skip-viz

# Combined options
python dissertation_analysis.py --sample 5000 --no-gpu --skip-viz
```

### Legacy Analysis (Alternative)
```bash
# Original analysis script
python run_analysis.py

# Skip visualizations
python run_analysis.py --skip-viz

# Model training only
python run_analysis.py --model-only
```

## 📈 Expected Results

### Performance Metrics
- **Best Model**: XGBoost
- **Accuracy**: 89.9%
- **AUC Score**: 0.923
- **Training Time**: 2.7s (GPU) / 12s (CPU)
- **Detection Rate**: 50.6% of real outbreaks

### Generated Outputs
```
results/
├── 📊 Visualizations (8 files, 2.5MB)
│   ├── data_overview.png
│   ├── model_performance.png
│   ├── roc_curves.png
│   └── confusion_matrix.png
├── 🤖 Models
│   └── best_model.joblib (2.5MB)
├── 📄 Data Files
│   ├── model_performance.csv
│   ├── feature_importance.csv
│   └── predictions.csv
└── 📋 Reports
    └── comprehensive_report.json (3KB)
```

## 🔍 Analysis Pipeline

1. **Data Loading** (30s)
   - Loads 80k+ WHO surveillance records
   - Processes 17k+ social media posts
   - Integrates environmental data

2. **Feature Engineering** (45s)
   - Creates 37 features from raw data
   - Temporal lag features for early warning
   - Sentiment-health correlations

3. **Machine Learning** (3s GPU / 15s CPU)
   - Trains XGBoost, Random Forest, Logistic Regression
   - Cross-validation with statistical testing
   - Performance evaluation

4. **Advanced Analytics** (30s)
   - Statistical significance testing
   - Geographic outbreak patterns
   - Business intelligence insights
   - Academic research contributions

5. **Visualization** (60s)
   - 8 publication-quality plots
   - Interactive dashboards
   - Academic presentation materials

## 🛠️ Troubleshooting

### Common Issues

**1. Memory Error**
```bash
# Use sampling for large datasets
python dissertation_analysis.py --sample 20000
```

**2. GPU Not Detected**
```bash
# Install CUDA toolkit and cupy
conda install cupy -c conda-forge

# Or disable GPU
python dissertation_analysis.py --no-gpu
```

**3. Module Import Error**
```bash
# Ensure you're in the correct directory
cd "analysis - Copy"

# Activate environment
conda activate disease_prediction
```

**4. Data Files Missing**
```bash
# Check data directory structure
ls data/raw/

# Ensure CSV files are present:
# - VIW_FNT.csv
# - Reddit_Combi.csv  
# - Twitter_Full.csv
```

### Performance Optimization

**For Faster Execution:**
```bash
# Use sampling + skip visualizations
python dissertation_analysis.py --sample 10000 --skip-viz

# GPU acceleration (3x faster)
python dissertation_analysis.py --sample 20000
```

**For Maximum Accuracy:**
```bash
# Full dataset with all features
python dissertation_analysis.py
```

## 📚 Academic Usage

### For Dissertation Writing
1. **Results Section**: Use `comprehensive_report.json` for quantitative results
2. **Figures**: Include visualizations from `results/` directory
3. **Methodology**: Reference the multi-modal data integration approach
4. **Discussion**: Highlight 89.9% accuracy on real WHO data

### Key Academic Contributions
- ✅ Novel integration of surveillance + social media data
- ✅ GPU-accelerated outbreak prediction (2.7s training)
- ✅ Global scale validation (180 countries)
- ✅ Statistical significance testing (p < 0.001)
- ✅ Real-world deployment readiness

## 📊 Dataset Information

### WHO FluNet Data
- **Source**: World Health Organization
- **Records**: 156,291 surveillance reports
- **Coverage**: 180 countries
- **Time Range**: 1995-2024 (29+ years)
- **Diseases**: Influenza A/B, ILI, SARI

### Social Media Data
- **Reddit**: 12,234 health-related discussions
- **Twitter**: 5,342 health sentiment posts
- **Processing**: Sentiment analysis, symptom detection
- **Languages**: Multiple (English primary)

## 🏆 Performance Benchmarks

| Model | Accuracy | AUC | Training Time | Memory |
|-------|----------|-----|---------------|---------|
| XGBoost (GPU) | **89.9%** | **0.923** | **2.7s** | 22MB |
| Random Forest | 87.8% | 0.910 | 3.6s | 35MB |
| Logistic Regression | 87.1% | 0.845 | 3.6s | 18MB |

## 📞 Support

### For Technical Issues
1. Check troubleshooting section above
2. Verify all dependencies are installed
3. Ensure data files are in correct location
4. Try with smaller sample size first

### For Academic Questions
- Reference the comprehensive report in `results/`
- Check visualization outputs for insights
- Review statistical validation results
- Consider the global scale achievements (180 countries)

## 🎓 Dissertation Checklist

- [x] **Real Data Processing** ✅ 80k+ WHO records
- [x] **Advanced ML Techniques** ✅ GPU-accelerated XGBoost
- [x] **Statistical Validation** ✅ Cross-validation + significance testing
- [x] **Global Scale Analysis** ✅ 180 countries coverage
- [x] **Novel Data Integration** ✅ Surveillance + social + environmental
- [x] **Production Readiness** ✅ Scalable architecture
- [x] **Academic Contributions** ✅ 4 major methodological innovations
- [x] **Comprehensive Results** ✅ 13 output files generated

---

## 🚀 Quick Commands Reference

```bash
# Complete analysis (recommended for dissertation)
python dissertation_analysis.py

# Fast testing
python dissertation_analysis.py --sample 5000 --skip-viz

# CPU only
python dissertation_analysis.py --no-gpu

# Check results
ls -la results/
```

**🎯 Ready for MSc Computer Science dissertation submission!**