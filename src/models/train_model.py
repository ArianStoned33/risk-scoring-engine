#!/usr/bin/env python3
"""
Script de entrenamiento para el proyecto de scoring de riesgo crediticio.

Este es el script principal del pipeline que coordina la carga de datos,
procesamiento, ingeniería de características y entrenamiento del modelo.
"""

import logging
import sys
from pathlib import Path
import joblib
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

# Agregar el directorio src al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

from data.make_dataset import main as make_dataset_main
from features.build_features import FeatureEngineer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CreditRiskModel:
    """
    Clase principal para el entrenamiento del modelo de scoring de riesgo crediticio.
    
    Esta clase encapsula toda la lógica de entrenamiento, incluyendo:
    - Carga y procesamiento de datos
    - Ingeniería de características
    - Entrenamiento del modelo
    - Validación cruzada
    - Guardado del modelo
    """
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Inicializa el modelo de scoring de riesgo crediticio.
        
        Args:
            model_type (str): Tipo de modelo a usar ('logistic_regression' o 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self, data_path: str = "data/03_primary") -> None:
        """
        Carga y prepara los datos para el entrenamiento.
        
        Args:
            data_path (str): Ruta a los datos procesados
        """
        logger.info("Cargando y preparando datos para entrenamiento")
        
        # Verificar si los datos existen, si no, ejecutar el pipeline de datos
        data_file = Path(data_path) / "credit_data_processed.csv"
        if not data_file.exists():
            logger.warning("Datos no encontrados, ejecutando pipeline de datos...")
            make_dataset_main()
        
        # Cargar datos
        import pandas as pd
        df = pd.read_csv(data_file)
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Preprocesar características
        X, y = self.feature_engineer.preprocess_features(df, fit=True)
        
        # Dividir en train y test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Datos divididos - Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")
        
    def create_model(self) -> None:
        """
        Crea el modelo de acuerdo al tipo especificado.
        """
        logger.info(f"Creando modelo: {self.model_type}")
        
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
            
    def train(self) -> None:
        """
        Entrena el modelo con los datos preparados.
        """
        logger.info("Iniciando entrenamiento del modelo")
        
        if self.model is None:
            self.create_model()
        
        # Entrenar modelo
        self.model.fit(self.X_train, self.y_train)
        logger.info("Modelo entrenado exitosamente")
        
        # Evaluar en datos de entrenamiento
        train_pred = self.model.predict(self.X_train)
        train_proba = self.model.predict_proba(self.X_train)[:, 1]
        
        train_auc = roc_auc_score(self.y_train, train_proba)
        logger.info(f"AUC-ROC en entrenamiento: {train_auc:.4f}")
        
    def validate(self) -> dict:
        """
        Realiza validación cruzada y evaluación en datos de prueba.
        
        Returns:
            dict: Diccionario con las métricas de evaluación
        """
        logger.info("Realizando validación del modelo")
        
        # Validación cruzada
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, 
            cv=5, scoring='roc_auc'
        )
        logger.info(f"Validación cruzada AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Evaluación en datos de prueba
        test_pred = self.model.predict(self.X_test)
        test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        test_auc = roc_auc_score(self.y_test, test_proba)
        logger.info(f"AUC-ROC en prueba: {test_auc:.4f}")
        
        # Reporte de clasificación
        report = classification_report(self.y_test, test_pred, output_dict=True)
        logger.info("Reporte de clasificación:\n" + str(report))
        
        return {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'test_auc': test_auc,
            'classification_report': report
        }
        
    def save_model(self, model_path: str = "models") -> None:
        """
        Guarda el modelo entrenado y el pipeline de features.
        
        Args:
            model_path (str): Directorio donde se guardarán los modelos
        """
        logger.info(f"Guardando modelo en: {model_path}")
        
        # Crear directorio si no existe
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo
        model_file = Path(model_path) / f"credit_risk_model_{self.model_type}.pkl"
        joblib.dump(self.model, model_file)
        
        # Guardar pipeline de features
        feature_pipeline_file = Path(model_path) / "feature_pipeline.pkl"
        joblib.dump(self.feature_engineer.pipeline, feature_pipeline_file)
        
        logger.info(f"Modelo guardado exitosamente: {model_file}")
        logger.info(f"Pipeline de features guardado: {feature_pipeline_file}")


def train_model(model_type: str = 'logistic_regression') -> CreditRiskModel:
    """
    Función principal de entrenamiento.
    
    Args:
        model_type (str): Tipo de modelo a entrenar
        
    Returns:
        CreditRiskModel: Modelo entrenado
    """
    logger.info("=== INICIANDO PIPELINE DE ENTRENAMIENTO ===")
    
    # Crear instancia del modelo
    credit_model = CreditRiskModel(model_type=model_type)
    
    # Cargar y preparar datos
    credit_model.load_and_prepare_data()
    
    # Entrenar modelo
    credit_model.train()
    
    # Validar modelo
    metrics = credit_model.validate()
    
    # Guardar modelo
    credit_model.save_model()
    
    logger.info("=== PIPELINE DE ENTRENAMIENTO COMPLETADO ===")
    
    return credit_model


def main():
    """Función principal ejecutable desde línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo de scoring crediticio')
    parser.add_argument('--model-type', '-m', default='logistic_regression',
                        choices=['logistic_regression', 'random_forest'],
                        help='Tipo de modelo a entrenar')
    parser.add_argument('--data-path', '-d', default='data/03_primary',
                        help='Ruta a los datos procesados')
    parser.add_argument('--model-path', '-o', default='models',
                        help='Directorio para guardar el modelo')
    
    args = parser.parse_args()
    
    try:
        model = train_model(model_type=args.model_type)
        logger.info("Entrenamiento completado exitosamente")
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()