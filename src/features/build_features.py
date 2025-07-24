#!/usr/bin/env python3
"""
Script de ingeniería de características para el proyecto de scoring de riesgo crediticio.

Este script toma los datos procesados desde data/03_primary y aplica transformaciones
de características para preparar los datos para el entrenamiento del modelo.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SimpleImputer
from sklearn.compose import ColumnTransformer
import argparse

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Clase para la ingeniería de características del modelo de scoring crediticio.
    
    Esta clase encapsula toda la lógica de preprocesamiento y transformación
    de características para el modelo.
    """
    
    def __init__(self):
        """Inicializa el FeatureEngineer con el pipeline de transformaciones."""
        self.pipeline = None
        self.feature_names = None
        
    def create_feature_pipeline(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Crea el pipeline de transformación de características.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            ColumnTransformer: Pipeline configurado para transformar las características
        """
        logger.info("Creando pipeline de transformación de características")
        
        # Identificar columnas numéricas (excluyendo target y ID)
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features 
                          if col not in ['id_cliente', 'default']]
        
        # Identificar columnas categóricas
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Pipeline para características numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Pipeline para características categóricas
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            # TODO: Agregar OneHotEncoder u otras transformaciones categóricas
        ])
        
        # Combinar transformadores
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        logger.info(f"Pipeline creado con {len(numeric_features)} features numéricas "
                   f"y {len(categorical_features)} features categóricas")
        
        return preprocessor
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica ingeniería de características al DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame con datos crudos
            
        Returns:
            pd.DataFrame: DataFrame con nuevas características ingenieradas
        """
        logger.info("Aplicando ingeniería de características")
        
        # TODO: Implementar ingeniería de características real
        # Ejemplos de features que podrían crearse:
        
        # 1. Ratio deuda/ingreso
        if 'deuda_actual' in df.columns and 'ingreso_anual' in df.columns:
            df['ratio_deuda_ingreso'] = df['deuda_actual'] / df['ingreso_anual']
        
        # 2. Categorías de edad
        if 'edad' in df.columns:
            df['categoria_edad'] = pd.cut(
                df['edad'], 
                bins=[0, 25, 35, 45, 55, 100],
                labels=['<25', '25-35', '35-45', '45-55', '55+']
            )
        
        # 3. Logaritmo de ingresos
        if 'ingreso_anual' in df.columns:
            df['log_ingreso'] = np.log1p(df['ingreso_anual'])
        
        # 4. Features de interacción
        if 'edad' in df.columns and 'ingreso_anual' in df.columns:
            df['edad_ingreso'] = df['edad'] * df['ingreso_anual']
        
        logger.info(f"Features creadas: {list(df.columns)}")
        return df
    
    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> tuple:
        """
        Preprocesa las características para el entrenamiento.
        
        Args:
            df (pd.DataFrame): DataFrame con las características
            fit (bool): Si True, ajusta el pipeline. Si False, solo transforma
            
        Returns:
            tuple: (X_transformed, y) donde X_transformed es el array preprocesado
                   y y es el vector objetivo
        """
        logger.info("Preprocesando características")
        
        # Separar features y target
        if 'default' not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'default'")
        
        X = df.drop(['default', 'id_cliente'], axis=1, errors='ignore')
        y = df['default']
        
        # Aplicar ingeniería de características
        X_engineered = self.engineer_features(X)
        
        # Crear y ajustar/transformar pipeline
        if self.pipeline is None or fit:
            self.pipeline = self.create_feature_pipeline(X_engineered)
        
        if fit:
            X_transformed = self.pipeline.fit_transform(X_engineered)
            logger.info("Pipeline ajustado y transformación aplicada")
        else:
            X_transformed = self.pipeline.transform(X_engineered)
            logger.info("Transformación aplicada con pipeline existente")
        
        # Guardar nombres de features (para debugging)
        self.feature_names = self._get_feature_names()
        
        return X_transformed, y
    
    def _get_feature_names(self) -> list:
        """
        Obtiene los nombres de las características después de la transformación.
        
        Returns:
            list: Lista con los nombres de las características transformadas
        """
        # TODO: Implementar lógica para obtener nombres de features
        # Esto es complejo con ColumnTransformer, por ahora retornamos índices
        return [f"feature_{i}" for i in range(100)]  # Placeholder
    
    def save_pipeline(self, filepath: str) -> None:
        """
        Guarda el pipeline de transformación.
        
        Args:
            filepath (str): Ruta donde guardar el pipeline
        """
        # TODO: Implementar guardado del pipeline
        logger.info(f"Pipeline guardado en: {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """
        Carga un pipeline previamente guardado.
        
        Args:
            filepath (str): Ruta del pipeline a cargar
        """
        # TODO: Implementar carga del pipeline
        logger.info(f"Pipeline cargado desde: {filepath}")


def main(input_path: str = "data/03_primary", output_path: str = "data/04_features") -> None:
    """
    Función principal que orquesta el proceso de ingeniería de características.
    
    Args:
        input_path (str): Ruta a los datos procesados
        output_path (str): Ruta donde se guardarán las características
    """
    logger.info("Iniciando proceso de ingeniería de características")
    
    try:
        # Cargar datos
        input_file = Path(input_path) / "credit_data_processed.csv"
        df = pd.read_csv(input_file)
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Inicializar FeatureEngineer
        fe = FeatureEngineer()
        
        # Preprocesar características
        X, y = fe.preprocess_features(df, fit=True)
        
        # Guardar datos transformados
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Guardar arrays como archivos numpy
        np.save(Path(output_path) / "X_features.npy", X)
        np.save(Path(output_path) / "y_target.npy", y)
        
        logger.info(f"Características guardadas en: {output_path}")
        logger.info(f"Shape de X: {X.shape}, Shape de y: {y.shape}")
        
    except Exception as e:
        logger.error(f"Error en el proceso de features: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingeniería de características')
    parser.add_argument('--input', '-i', default='data/03_primary',
                        help='Ruta a los datos procesados')
    parser.add_argument('--output', '-o', default='data/04_features',
                        help='Ruta de salida para las características')
    
    args = parser.parse_args()
    main(args.input, args.output)