# Real Estate Price Prediction System

A machine learning system for accurate real estate price prediction with cloud-based deployment and real-time API access.

## Overview

This project implements an end-to-end machine learning solution for predicting residential property prices using comprehensive feature engineering, advanced modeling techniques, and cloud infrastructure. The system combines traditional statistical methods with modern ML algorithms to provide accurate, interpretable price predictions.

## Features

- **Accurate Predictions**: Achieves 85%+ accuracy on property valuations
- **Real-time API**: RESTful API for instant price predictions
- **Interactive Dashboard**: Web-based interface for data exploration and predictions
- **Cloud Deployment**: Scalable AWS infrastructure with automated deployment
- **Model Interpretability**: SHAP-based explanations for prediction transparency

## Architecture

```
Data Sources → AWS S3 → Lambda Functions → API Gateway → Web Interface
                ↓
            DynamoDB ← CloudWatch Monitoring
```

## Technology Stack

### Core ML Stack
- **Python 3.9+**: Primary development language
- **scikit-learn**: Machine learning algorithms
- **pandas & numpy**: Data manipulation and analysis
- **XGBoost**: Gradient boosting for optimal performance

### Cloud Infrastructure
- **AWS EC2**: Model training and batch processing
- **AWS Lambda**: Serverless prediction functions
- **AWS S3**: Data and model storage
- **AWS API Gateway**: RESTful API endpoints
- **AWS DynamoDB**: Prediction logging and monitoring

### Web Interface
- **Streamlit**: Interactive dashboard
- **Plotly**: Dynamic visualizations

### DevOps & Deployment
- **GitHub Actions**: CI/CD pipeline
- **CloudFormation**: Infrastructure as code
- **AWS CloudWatch**: Monitoring and alerting

## Quick Start

### Prerequisites
- Python 3.9 or higher
- AWS 
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/usama-rao/real-estate-ml-project.git
   cd real-estate-ml-project
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AWS credentials**
   ```bash
   aws configure
   ```

### Local Development

1. **Train the model**
   ```bash
   python src/models/train_model.py
   ```

2. **Start the API server**
   ```bash
   python src/api/app.py
   ```

3. **Launch the dashboard**
   ```bash
   streamlit run src/dashboard/app.py
   ```

### Making Predictions

#### API Usage
```bash
curl -X POST https://your-api-endpoint.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sqft": 2000,
    "bedrooms": 3,
    "bathrooms": 2,
    "neighborhood": "Downtown",
    "year_built": 1995
  }'
```

#### Python Client
```python
from src.models.predictor import RealEstatePredictor

predictor = RealEstatePredictor()
price = predictor.predict({
    'sqft': 2000,
    'bedrooms': 3,
    'bathrooms': 2,
    'neighborhood': 'Downtown',
    'year_built': 1995
})
print(f"Predicted Price: ${price:,.2f}")
```

## Model Performance

| Metric | Value |
|--------|-------|
| Mean Absolute Error | $21,500 |
| Root Mean Square Error | $31,800 |
| R² Score | 0.892 |
| Mean Absolute Percentage Error | 8.7% |

### Model Comparison

| Algorithm | MAE | RMSE | R² | Training Time |
|-----------|-----|------|----|--------------| 
| Linear Regression | $35,200 | $48,100 | 0.754 | 0.1s |
| Random Forest | $23,100 | $33,900 | 0.871 | 2.3s |
| XGBoost | $21,500 | $31,800 | 0.892 | 1.8s |

## API Documentation

### Endpoints

#### POST /predict
Predict property price based on features.

**Request Body:**
```json
{
  "sqft": 2000,
  "bedrooms": 3,
  "bathrooms": 2,
  "neighborhood": "Downtown",
  "year_built": 1995,
  "lot_size": 0.25,
  "garage_spaces": 2
}
```

**Response:**
```json
{
  "predicted_price": 285000,
  "confidence_interval": [265000, 305000],
  "model_version": "v1.2.0",
  "prediction_id": "uuid-string"
}
```

#### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "v1.2.0",
  "timestamp": "2025-01-31T10:30:00Z"
}
```

## Deployment

### AWS Deployment

1. **Deploy infrastructure**
   ```bash
   aws cloudformation deploy --template-file deployment/infrastructure.yaml \
     --stack-name real-estate-ml --capabilities CAPABILITY_IAM
   ```

2. **Deploy application**
   ```bash
   ./deployment/deploy.sh
   ```

## Project Structure

```
real-estate-ml-project/
├── src/
│   ├── data/           # Data processing modules
│   ├── features/       # Feature engineering
│   ├── models/         # ML models and training
│   ├── api/           # API endpoints
│   └── dashboard/     # Streamlit dashboard
├── tests/             # Unit and integration tests
├── deployment/        # Cloud deployment configs
├── docs/             # Technical documentation
├── models/           # Trained model artifacts
└── requirements.txt  # Python dependencies
```

## Contributing

This project follows standard development practices:

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or collaboration opportunities:

- **GitHub**: https://github.com/usama-rao
- **LinkedIn**: https://www.linkedin.com/in/usamatariqrao/
- **Email**: usamatariq.workmail@gmail.com

## Acknowledgments

- Kaggle community for datasets
- AWS for cloud infrastructure
- Open source ML community for tools and libraries