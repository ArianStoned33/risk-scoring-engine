#!/usr/bin/env python3
"""
Script de entrenamiento para el proyecto de scoring de riesgo crediticio.
"""

import logging
import sys
from pathlib import Path
import joblib
import argparse
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CreditRiskModel:
    """
    Clase para el entrenamiento del modelo de scoring de riesgo crediticio.
    """
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        
    def load_data(self, data_path: str = "data/04_features") -> None:
        """
        Carga las características y el target preprocesados.
        """
        logger.info(f"Cargando datos desde: {data_path}")
        data_path = Path(data_path)
        
        self.X = np.load(data_path / "X_features.npy", allow_pickle=True)
        self.y = np.load(data_path / "y_target.npy", allow_pickle=True)
        
        # Dividir en train y test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        logger.info(f"Datos divididos - Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")

    def create_model(self) -> None:
        """Crea el modelo según el tipo especificado."""
        logger.info(f"Creando modelo: {self.model_type}")
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
            
    def train(self) -> None:
        """Entrena el modelo con los datos preparados."""
        logger.info("Iniciando entrenamiento del modelo")
        if self.model is None:
            self.create_model()
        
        self.model.fit(self.X_train, self.y_train)
        logger.info("Modelo entrenado exitosamente")
        
    def validate(self) -> dict:
        """Realiza validación y evaluación del modelo."""
        logger.info("Realizando validación del modelo")
        
        # Validación cruzada
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        logger.info(f"Validación cruzada AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Evaluación en datos de prueba
        test_proba = self.model.predict_proba(self.X_test)[:, 1]
        test_auc = roc_auc_score(self.y_test, test_proba)
        logger.info(f"AUC-ROC en prueba: {test_auc:.4f}")
        
        return {'cv_auc_mean': cv_scores.mean(), 'test_auc': test_auc}
        
    def save_model(self, model_path: str = "models") -> None:
        """Guarda el modelo entrenado."""
        logger.info(f"Guardando modelo en: {model_path}")
        output_dir = Path(model_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = output_dir / f"credit_risk_model_{self.model_type}.pkl"
        joblib.dump(self.model, model_file)
        logger.info(f"Modelo guardado exitosamente: {model_file}")


def train_model(params: dict) -> None:
    """
    Función principal de entrenamiento.
    """
    logger.info("=== INICIANDO PIPELINE DE ENTRENAMIENTO ===")
    
    model_type = params['models']['model_type']
    
    credit_model = CreditRiskModel(model_type=model_type)
    credit_model.load_data()
    credit_model.train()
    credit_model.validate()
    credit_model.save_model()
    
    logger.info("=== PIPELINE DE ENTRENAMIENTO COMPLETADO ===")


if __name__ == "__main__":
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
        
    train_model(params=params)