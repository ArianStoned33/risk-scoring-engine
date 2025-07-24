#!/usr/bin/env python3
"""
Script de procesamiento de datos para el proyecto de scoring de riesgo crediticio.

Este script se encarga de cargar los datos crudos desde data/01_raw,
aplicar transformaciones básicas y guardar los datos procesados en data/03_primary.
"""

import logging
import pandas as pd
from pathlib import Path
import argparse

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(input_path: str) -> pd.DataFrame:
    """
    Carga los datos desde el directorio especificado.
    
    Args:
        input_path (str): Ruta al directorio que contiene los datos crudos
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
        
    Note:
        Esta función es un marcador de posición. En producción, deberá:
        - Validar la existencia de archivos
        - Soportar múltiples formatos (CSV, Excel, JSON, etc.)
        - Manejar errores de lectura
    """
    logger.info(f"Cargando datos desde: {input_path}")
    
    # TODO: Implementar lógica real de carga de datos
    # Por ahora, generamos datos de ejemplo
    data = {
        'id_cliente': range(1, 101),
        'edad': [25, 35, 45, 30, 40] * 20,
        'ingreso_anual': [30000, 50000, 75000, 40000, 60000] * 20,
        'historial_crediticio': [1, 0, 1, 0, 1] * 20,
        'deuda_actual': [5000, 10000, 15000, 8000, 12000] * 20,
        'default': [0, 1, 0, 1, 0] * 20
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Guarda el DataFrame procesado en el directorio especificado.
    
    Args:
        df (pd.DataFrame): DataFrame a guardar
        output_path (str): Ruta donde se guardarán los datos
        
    Note:
        Esta función es un marcador de posición. En producción, deberá:
        - Crear directorios si no existen
        - Validar permisos de escritura
        - Soportar diferentes formatos de salida
    """
    logger.info(f"Guardando datos procesados en: {output_path}")
    
    # Crear directorio si no existe
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Guardar como CSV
    output_file = Path(output_path) / "credit_data_processed.csv"
    df.to_csv(output_file, index=False)
    
    logger.info(f"Datos guardados exitosamente en: {output_file}")


def main(input_path: str = "data/01_raw", output_path: str = "data/03_primary") -> None:
    """
    Función principal que orquesta el proceso de carga y guardado de datos.
    
    Args:
        input_path (str): Ruta a los datos crudos
        output_path (str): Ruta donde se guardarán los datos procesados
    """
    logger.info("Iniciando proceso de procesamiento de datos")
    
    try:
        # Cargar datos
        df = load_data(input_path)
        
        # TODO: Agregar validaciones de datos
        # TODO: Agregar limpieza de datos
        # TODO: Agregar transformaciones básicas
        
        # Guardar datos procesados
        save_data(df, output_path)
        
        logger.info("Proceso de procesamiento de datos completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en el proceso de datos: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Procesamiento de datos de crédito')
    parser.add_argument('--input', '-i', default='data/01_raw',
                        help='Ruta a los datos crudos')
    parser.add_argument('--output', '-o', default='data/03_primary',
                        help='Ruta de salida para datos procesados')
    
    args = parser.parse_args()
    main(args.input, args.output)