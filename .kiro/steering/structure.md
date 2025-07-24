# Project Structure & Organization

## Directory Layout

```
/
├── .github/              # CI/CD workflows and GitHub Actions
├── .kiro/                # Kiro AI assistant configuration and steering
├── data/                 # Data storage (not version controlled)
│   ├── 01_raw/          # Original, immutable data from sources
│   ├── 02_intermediate/ # Partially processed data
│   └── 03_primary/      # Final, analysis-ready datasets
├── docs/                 # Project documentation
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code modules
│   ├── api/             # FastAPI application and endpoints
│   ├── data/            # Data processing and ingestion scripts
│   ├── features/        # Feature engineering modules
│   └── models/          # Model training, evaluation, and prediction
├── tests/               # Unit and integration tests
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
└── README.md           # Project overview and setup instructions
```

## Code Organization Principles

### Modular Architecture
- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Loose Coupling**: Modules interact through well-defined interfaces
- **High Cohesion**: Related functionality is grouped together
- **Reusability**: Components can be used across different parts of the system

### Data Flow Architecture
1. **Raw Data** (`data/01_raw/`) → Immutable source data
2. **Intermediate Data** (`data/02_intermediate/`) → Cleaned and validated data
3. **Primary Data** (`data/03_primary/`) → Analysis-ready datasets
4. **Features** → Engineered features for model training
5. **Models** → Trained model artifacts and metadata

### Source Code Structure

#### `src/data/` - Data Processing Layer
- **Purpose**: Data ingestion, cleaning, and validation
- **Key Files**:
  - `make_dataset.py`: Main data processing pipeline
  - `ingestion.py`: Data loading from various sources
  - `preprocessing.py`: Data cleaning and transformation
  - `validation.py`: Data quality checks and validation

#### `src/features/` - Feature Engineering Layer
- **Purpose**: Transform raw data into ML-ready features
- **Key Files**:
  - `build_features.py`: Main feature engineering pipeline
  - **Pattern**: Use FeatureEngineer class with fit/transform methods
  - **Output**: Preprocessed feature matrices and pipelines

#### `src/models/` - Machine Learning Layer
- **Purpose**: Model training, evaluation, and prediction
- **Key Files**:
  - `train_model.py`: Main training pipeline
  - `evaluate_model.py`: Model evaluation and metrics
  - `predict_model.py`: Inference and prediction logic
- **Pattern**: Use BaseModel abstract class for consistent interfaces

#### `src/api/` - API Service Layer
- **Purpose**: REST API for real-time scoring
- **Key Files**:
  - `main.py`: FastAPI application setup
  - `models.py`: Pydantic data models for validation
  - `scoring.py`: Business logic for risk scoring
  - `middleware.py`: Logging, authentication, rate limiting

## Naming Conventions

### Files and Directories
- **Snake_case**: Use lowercase with underscores (e.g., `make_dataset.py`)
- **Descriptive Names**: File names should clearly indicate their purpose
- **Module Structure**: Group related functionality in packages with `__init__.py`

### Python Code
- **Classes**: PascalCase (e.g., `CreditRiskModel`, `FeatureEngineer`)
- **Functions/Variables**: snake_case (e.g., `train_model`, `feature_pipeline`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MODEL_VERSION`, `DEFAULT_THRESHOLD`)
- **Private Methods**: Leading underscore (e.g., `_validate_input`)

### Configuration and Environment
- **Environment Variables**: UPPER_SNAKE_CASE (e.g., `GOOGLE_CLOUD_PROJECT`)
- **Config Files**: Use descriptive names (e.g., `.env.example`, `requirements.txt`)

## Import Patterns

### Standard Import Order
1. Standard library imports
2. Third-party library imports
3. Local application imports

### Example Import Structure
```python
# Standard library
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Local imports
from src.data.make_dataset import load_data
from src.features.build_features import FeatureEngineer
```

## Error Handling Patterns

### Logging Strategy
- **Structured Logging**: Use consistent format across all modules
- **Log Levels**: INFO for normal operations, WARNING for recoverable issues, ERROR for failures
- **Context**: Include relevant context (file paths, data shapes, model metrics)

### Exception Handling
- **Specific Exceptions**: Catch specific exceptions rather than broad Exception
- **Graceful Degradation**: Provide fallback behavior when possible
- **User-Friendly Messages**: Log technical details, return user-friendly error messages

## Testing Structure

### Test Organization
- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test component interactions and data flow
- **End-to-End Tests**: Test complete workflows from data to prediction

### Test File Naming
- **Pattern**: `test_<module_name>.py` (e.g., `test_train_model.py`)
- **Location**: Mirror source structure in `tests/` directory
- **Fixtures**: Use pytest fixtures for common test data and setup

## Documentation Standards

### Code Documentation
- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Type Hints**: Include type annotations for function parameters and returns
- **Comments**: Explain complex business logic and algorithmic decisions

### Project Documentation
- **README.md**: Project overview, setup instructions, and usage examples
- **docs/**: Detailed technical documentation and architecture decisions
- **Inline Comments**: Explain non-obvious code sections and business rules