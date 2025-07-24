# Multi-stage Dockerfile para optimizar tamaño de imagen

# Etapa 1: Builder - Instalación de dependencias
FROM python:3.9-slim-buster as builder

# Establecer variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiar archivo de requisitos
COPY requirements.txt .

# Instalar dependencias en el entorno virtual
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Etapa 2: Imagen final - Solo runtime
FROM python:3.9-slim-buster

# Establecer variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Crear usuario no-root para seguridad
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copiar entorno virtual desde la etapa builder
COPY --from=builder /opt/venv /opt/venv

# Establecer directorio de trabajo
WORKDIR /app

# Copiar código fuente
COPY src/ ./src/

# Cambiar propietario de los archivos al usuario no-root
RUN chown -R appuser:appuser /app

# Cambiar al usuario no-root
USER appuser

# Establecer comando por defecto para ejecutar el entrenamiento
CMD ["python", "src/models/train_model.py"]