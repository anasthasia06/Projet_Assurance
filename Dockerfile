FROM python:3.11-slim

WORKDIR /app

# Copie et installation des dépendances d'abord (optimise le cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source et des modèles
COPY src/ ./src/
COPY models/ ./models/

# Expose le port de l'API
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
