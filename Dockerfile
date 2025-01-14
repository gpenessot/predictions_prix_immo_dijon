# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Installation de uv
RUN pip install uv

# Copie des fichiers de configuration
COPY pyproject.toml .

# Installation des dépendances avec uv
RUN uv pip install .

# Copie du code source
COPY api/ api/
COPY models/ models/
COPY src/ src/

# Port par défaut pour FastAPI
EXPOSE 8000

# Commande par défaut
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
