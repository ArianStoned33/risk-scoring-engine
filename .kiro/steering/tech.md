# Technology Stack & Build System

## Core Technology Stack

### Programming Language & Libraries
- **Python 3.9+**: Primary development language
- **Data Science**: pandas>=2.0.0, numpy>=1.24.0, scikit-learn>=1.3.0
- **ML Models**: XGBoost, LightGBM for gradient boosting
- **API Framework**: FastAPI>=0.100.0 with uvicorn for high-performance REST API
- **Validation**: Pydantic>=2.0.0 for data validation and serialization

### Cloud Platform (Google Cloud Platform)
- **Storage**: Google Cloud Storage for data lakes and model artifacts
- **ML Platform**: Vertex AI for training, model registry, and monitoring
- **Deployment**: Cloud Run for serverless API hosting
- **Orchestration**: Vertex AI Pipelines for ML workflow automation
- **Monitoring**: Vertex AI Model Monitoring for drift detection

### Development & Deployment
- **Containerization**: Docker with multi-stage builds for optimization
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Environment Management**: python-dotenv for configuration
- **Model Tracking**: Vertex AI Experiments for experiment management

### Dashboard & Monitoring
- **Visualization**: Streamlit for interactive business dashboards
- **Logging**: Python logging with structured output
- **Metrics**: Custom performance tracking and alerting

## Common Commands

### Development Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your GCP credentials
```

### Data Pipeline
```bash
# Process raw data
python src/data/make_dataset.py --input data/01_raw --output data/03_primary

# Build features
python src/features/build_features.py --input data/03_primary --output data/04_features
```

### Model Training
```bash
# Train logistic regression model
python src/models/train_model.py --model-type logistic_regression

# Train random forest model
python src/models/train_model.py --model-type random_forest

# Evaluate model
python src/models/evaluate_model.py --model-path models/
```

### API Development
```bash
# Run API locally
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Test API endpoints
curl -X POST "http://localhost:8000/score" -H "Content-Type: application/json" -d @test_data.json
```

### Docker Operations
```bash
# Build Docker image
docker build -t credit-risk-api .

# Run container locally
docker run -p 8000:8000 --env-file .env credit-risk-api

# Multi-stage build for production
docker build --target production -t credit-risk-api:prod .
```

### Testing
```bash
# Run unit tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run integration tests
python -m pytest tests/integration/ -v
```

## Code Quality Standards
- **Logging**: Use structured logging with appropriate levels (INFO, WARNING, ERROR)
- **Error Handling**: Implement comprehensive try-catch blocks with meaningful error messages
- **Type Hints**: Use Python type hints for better code documentation
- **Documentation**: Include docstrings for all classes and functions
- **Configuration**: Use environment variables for all configurable parameters