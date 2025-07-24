Motor de Scoring de Riesgo Crediticio End-to-End 🏦

## 1. Visión General

Este repositorio contiene un proyecto de nivel profesional que demuestra la construcción de un sistema de Machine Learning de extremo a extremo para el scoring de riesgo crediticio. El objetivo es simular un entorno de producción real, aplicando las mejores prácticas de **Ingeniería de Machine Learning (MLOps)** y **Arquitectura de Sistemas de ML en la Nube**.

El proyecto está diseñado para ser una pieza central de un portafolio, alineado con las habilidades más demandadas por la industria para roles de **Senior ML Engineer** y **Arquitecto de ML**.

Para una inmersión profunda en la arquitectura, los requerimientos y el plan de implementación, por favor consulte la documentación detallada en la carpeta `/docs`:
-   **[`docs/design.md`](docs/design.md):** Documento de Diseño y Arquitectura Técnica.
-   **[`docs/requirements.md`](docs/requirements.md):** Requerimientos de Producto y Criterios de Aceptación.
-   **[`docs/tasks.md`](docs/tasks.md):** Plan de Implementación y Checklist de Tareas.
-   **[`docs/implementation_notes.md`](docs/implementation_notes.md):** Notas Técnicas para el Desarrollo.

## 2. Stack Tecnológico Principal

Este proyecto utiliza un stack tecnológico moderno, pragmático y basado en servicios gestionados de **Google Cloud Platform (GCP)** para maximizar la eficiencia y la escalabilidad.

-   **Lenguaje y Librerías:** Python, Pandas, Scikit-Learn, XGBoost/LightGBM
-   **API:** FastAPI
-   **Cloud Provider:** Google Cloud Platform (GCP)
-   **Contenerización:** Docker
-   **CI/CD:** GitHub Actions
-   **Orquestación de Pipelines:** Vertex AI Pipelines
-   **Registro y Tracking:** Vertex AI Model Registry & Experiments
-   **Despliegue Serverless:** Cloud Run
-   **Monitoreo de Modelos:** Vertex AI Model Monitoring
-   **Dashboarding:** Streamlit

## 3. Estructura del Proyecto

La estructura del proyecto es modular y está diseñada para la escalabilidad y el mantenimiento, separando claramente las responsabilidades:

```
/
├── .github/              # Workflows de CI/CD con GitHub Actions.
├── data/                 # Datos del proyecto (no versionados en Git).
├── docs/                 # Documentación clave del proyecto.
├── notebooks/            # Jupyter notebooks para análisis exploratorio.
├── src/                  # Código fuente principal de la aplicación.
│   ├── api/              # Código para la API de inferencia (FastAPI).
│   ├── data/             # Scripts para el procesamiento de datos.
│   ├── features/         # Scripts para la ingeniería de características.
│   ├── models/           # Scripts para entrenar y evaluar modelos.
│   └── monitoring/       # Scripts para el monitoreo de drift.
├── tests/                # Pruebas unitarias y de integración.
├── Dockerfile            # Define la imagen Docker para producción.
└── requirements.txt      # Dependencias de Python.
```

## 4. Instalación y Uso

1.  **Clonar el repositorio:**
    ```bash
    git clone <https://github.com/ArianStoned33/risk-scoring-engine.git>
    cd risk-scoring-engine
    ```

2.  **Crear un entorno virtual e instalar dependencias:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configurar variables de entorno:**
    Crea un archivo `.env` a partir de `.env.example` y rellena las variables necesarias (credenciales de GCP, nombre del bucket, etc.).

4.  **Ejecutar el pipeline de entrenamiento (Ejemplo):**
    Los flujos de trabajo se ejecutan a través de scripts en `src/`. Para entrenar un modelo:
    ```bash
    python src/models/train_model.py
