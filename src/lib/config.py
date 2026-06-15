from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
import logging
import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    cors_origins: str = Field(
        default="*",
        description="Comma-separated allowed origins, or * for any.",
    )
    app_name: str = "Dog Breed Recognition TP2"

    # Modelo de embeddings seleccionado: baseline | resnet18_finetuned | cnn_custom
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "baseline")
    model_path: Path = Path(os.getenv("MODEL_PATH", "models"))
    resnet18_model_name: str = os.getenv("RESNET18_MODEL_NAME", "resnet18_finetuned.pth")
    cnn_custom_model_name: str = os.getenv("CNN_CUSTOM_MODEL_NAME", "cnn_custom.pth")

    # Busqueda por similitud
    similarity_metric: str = os.getenv("SIMILARITY_METRIC", "cosine")
    similarity_threshold: float = os.getenv("SIMILARITY_THRESHOLD", 0.55)
    top_k: int = os.getenv("TOP_K", 10)
    image_size: int = os.getenv("IMAGE_SIZE", 224)
    embedding_dim: int = os.getenv("EMBEDDING_DIM", 512)

    # Configuracion de YOLO (Etapa 3)
    yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    yolo_conf_threshold: float = os.getenv("YOLO_CONF_THRESHOLD", 0.25)
    yolo_dog_class_id: int = os.getenv("YOLO_DOG_CLASS_ID", 16)

    # Paths
    embeddings_path: Path = Path(os.getenv("EMBEDDINGS_PATH", "data/embeddings.json"))
    data_path: Path = Path(os.getenv("DATA_PATH", "data"))
    dataset_path: Path = Path(os.getenv("DATASET_PATH", "data/dataset"))
    output_path: Path = Path(os.getenv("OUTPUT_PATH", "output"))
    max_workers: int = os.getenv("MAX_WORKERS", 2)

    # PostgreSQL / pgvector
    use_pgvector: bool = os.getenv("USE_PGVECTOR", "True").lower() == "true"
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = os.getenv("POSTGRES_PORT", 5432)
    postgres_db: str = os.getenv("POSTGRES_DB", "dogs")
    postgres_user: str = os.getenv("POSTGRES_USER", "dogs_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "dogs_pass")


# ImageNet normalization constants (mean/std por canal)
# Usados por similarity_service y classifier_service para
# transforms.Normalize()
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
