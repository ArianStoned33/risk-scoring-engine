#!/usr/bin/env python3
"""
API de inferencia para el modelo de scoring de riesgo crediticio.
"""

import logging
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="API para predecir el riesgo de default crediticio.",
    version="0.1.0"
)

# --- Modelos de Datos (Pydantic) ---
class ClientData(BaseModel):
    """Define la estructura de los datos de entrada para un cliente."""
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    # Añade aquí cualquier otra feature que tu modelo espere

class PredictionResponse(BaseModel):
    """Define la estructura de la respuesta de la predicción."""
    prediction: int
    probability: float
    risk_level: str

# --- Carga del Modelo ---
MODEL_DIR = Path("models")
PIPELINE_DIR = Path("data/04_features")

model = None
feature_pipeline = None

@app.on_event("startup")
async def startup_event():
    global model, feature_pipeline
    logger.info("Iniciando la API y cargando el modelo...")
    try:
        model_path = MODEL_DIR / "credit_risk_model_logistic_regression.pkl"
        pipeline_path = PIPELINE_DIR / "feature_pipeline.pkl"
        
        model = joblib.load(model_path)
        feature_pipeline = joblib.load(pipeline_path)
        
        logger.info("Modelo y pipeline de features cargados exitosamente.")
    except FileNotFoundError as e:
        logger.error(f"Error al cargar el modelo o el pipeline: {e}")
        # En un entorno de producción, podrías querer que la API no inicie si no puede cargar el modelo.
        model = None
        feature_pipeline = None

# --- Endpoints ---

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Endpoint para verificar que la API está funcionando.
    """
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/score", response_model=PredictionResponse, tags=["Scoring"])
async def predict_score(data: ClientData):
    """
    Realiza una predicción de riesgo crediticio para un cliente.
    """
    if not model or not feature_pipeline:
        return {"error": "Modelo no cargado. La API no está lista para predicciones."}

    logger.info(f"Recibida petición de scoring para: {data.dict()}")
    
    # Convertir datos de Pydantic a DataFrame de Pandas
    input_df = pd.DataFrame([data.dict()])
    
    # Aplicar el pipeline de preprocesamiento
    # Nota: El pipeline espera todas las columnas que usó en el entrenamiento.
    # Aquí simplificamos asumiendo que las columnas del DataFrame coinciden.
    # En un caso real, necesitarías asegurar que todas las columnas estén presentes.
    
    # Placeholder para columnas que el pipeline espera pero no vienen en la request
    # Esto es una simplificación y debería manejarse de forma más robusta
    for col in feature_pipeline.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0 # O un valor por defecto apropiado
            
    X_processed = feature_pipeline.transform(input_df)
    
    # Realizar predicción
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0, 1]
    
    # Determinar nivel de riesgo
    if probability < 0.3:
        risk_level = "Bajo"
    elif probability < 0.7:
        risk_level = "Medio"
    else:
        risk_level = "Alto"
        
    logger.info(f"Predicción: {prediction}, Probabilidad: {probability:.4f}")
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": risk_level
    }