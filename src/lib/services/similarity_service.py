from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np

import torch
import torchvision.models as models
from torchvision import transforms

from lib.config import IMAGENET_MEAN, IMAGENET_STD
from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)


class SimilarityService:
    """Etapa 1: buscador de imagenes por similitud.

    Funciones a implementar por el estudiante:
      - extract_embedding(image)
      - search_similar_images(embedding, top_k)
      - predict_breed_from_neighbors(results)

    La orquestacion (search, index_image, persistencia y metricas de similitud)
    ya esta provista y no debe modificarse sin justificarlo en el informe.
    """

    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        top_k: int,
        image_size: int,
        model_name: str,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.image_size = image_size
        self.model_name = model_name
        self.url_resolver = url_resolver

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV)
        return image

    # ------------------------------------------------------------------
    # Etapa 1: funciones a implementar
    # ------------------------------------------------------------------

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """Genera el embedding de una imagen usando ResNet18 pre-entrenado.

        Preprocesa la imagen (resize, normalizacion ImageNet), la pasa por el backbone
        del modelo (sin la capa de clasificacion) y retorna el embedding como lista de floats.
        """
        # ------------------------------------------------------------------
        # 1. Cargar el modelo pre-entrenado (solo la primera vez)
        # ------------------------------------------------------------------
        # Usamos hasattr para cachear el modelo como atributo de instancia.
        # La primera llamada lo carga de torchvision, las siguientes reusan
        # el mismo objeto en memoria sin recargar pesos ni reasignar device.
        if not hasattr(self, '_model'):
            # ResNet18 genera embeddings de 512 features (EMBEDDING_DIM=512).
            # Alternativas como ResNet50 darian 2048 features, incompatibles
            # con la dimension configurada en el esquema de la base vectorial.
            self._model = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )

            # El modelo completo tiene una capa fc (512->1000 clases) al final.
            # Para obtener embeddings descartamos fc con children()[:-1] y
            # envolvemos el resto en Sequential: el backbone incluye avgpool
            # que reduce el feature map a 1x1, dando un vector 512-D por imagen.
            self._model = torch.nn.Sequential(
                *list(self._model.children())[:-1]
            )

            # .eval() es critico: BatchNorm y Dropout se comportan distinto
            # en entrenamiento vs inferencia. Sin eval(), el embedding seria
            # inconsistente entre llamadas.
            self._model.eval()
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = self._model.to(self._device)

        # ------------------------------------------------------------------
        # 2. Pipeline de preprocesamiento
        # ------------------------------------------------------------------
        # El pipeline replica el preprocesamiento con el que los modelos de
        # torchvision fueron entrenados en ImageNet:
        #   - ToPILImage: convierte numpy (H,W,3) a PIL, necesario porque
        #     torchvision.transforms opera sobre PIL.
        #   - Resize: 224x224 es el tamano estandar de entrada de ResNet18.
        #   - ToTensor: reordena dims (H,W,C -> C,H,W) y escala [0,255]->[0,1].
        #   - Normalize: centra y escala con media y std de ImageNet para que
        #     los valores de entrada coincidan con la distribucion que el modelo
        #     "espera" (media ~0, std ~1).
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # ------------------------------------------------------------------
        # 3. BGR -> RGB, preprocesar, forward
        # ------------------------------------------------------------------
        # OpenCV carga en BGR por convencion historica; torchvision espera
        # RGB porque ImageNet esta en RGB. La conversion es obligatoria para
        # que los canales coincidan con los pesos pre-entrenados.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # unsqueeze(0) agrega la dimension de batch (1, C, H, W) porque el
        # modelo espera lotes de imagenes. to(device) mueve el tensor a GPU
        # si esta disponible.
        input_tensor = transform(rgb).unsqueeze(0).to(self._device)

        # no_grad() desactiva autograd: en inferencia no necesitamos gradientes,
        # lo que ahorra memoria y acelera la ejecucion.
        with torch.no_grad():
            # squeeze() elimina las dimensiones espaciales (1, 512, 1, 1) -> (512,)
            # dejando solo el vector de features. cpu().numpy() mueve a CPU y
            # convierte a numpy array para poder serializar como lista.
            emb = self._model(input_tensor).squeeze().cpu().numpy()

        # tolist() convierte numpy array a list[float], formato requerido por
        # la base vectorial y los schemas de la aplicacion.
        return emb.tolist()

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """Recupera de la base vectorial las top_k imagenes mas similares.

        Usa self.store.all() + self.similarity() para compatibilidad con ambos
        backends (pgvector y JSON). Retorna lista de Neighbor ordenada por score
        descendente.
        """
        # ------------------------------------------------------------------
        # Recuperacion y scoring
        # ------------------------------------------------------------------
        # Usamos self.store.all() para traer todos los registros y calcular
        # similitud en Python con self.similarity(). Este approach funciona
        # con ambos backends (pgvector_store y embedding_store JSON), a
        # diferencia de self.store.search() que tiene problemas de tipado
        # con pgvector (double precision[] vs vector). Para ~5000 imagenes
        # la diferencia de performance es imperceptible.
        all_records = self.store.all()

        # scored: lista de tuplas (score, record). self.similarity() usa la
        # metrica configurada en .env (cosine por defecto, l2 como alternativa).
        # Devuelve float en [0,1]: 1 = identico, ~0 = sin relacion.
        scored = [
            (self.similarity(embedding, r.embedding), r)
            for r in all_records
        ]

        # Orden descendente: mayor similitud primero
        sorted_scored = sorted(scored, key=lambda x: x[0], reverse=True)

        # Tomamos los top_k y convertimos al schema Neighbor que espera el
        # frontend. Redondeamos a 6 decimales para evitar falsos positivos
        # por diferencias de precision numerica.
        return [
            Neighbor(path=r.path, breed=r.breed, score=round(s, 6))
            for s, r in sorted_scored[:top_k]
        ]

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """Predice la raza por voto ponderado por score de los vecinos.

        Si el mejor score esta por debajo de self.similarity_threshold se
        retorna "unknown". Retorna (raza, score del mejor vecino).
        """
        # ------------------------------------------------------------------
        # Caso borde: sin vecinos no hay prediccion posible
        # ------------------------------------------------------------------
        if not results:
            return ("unknown", 0.0)

        # results viene ordenado por search_similar_images (descendente),
        # asi que el mejor score esta en la primera posicion.
        best_score = results[0].score

        # ------------------------------------------------------------------
        # Umbral de confianza (SIMILARITY_THRESHOLD)
        # ------------------------------------------------------------------
        # Si el vecino mas similar tiene un score por debajo del threshold,
        # significa que la imagen consultada no se parece a ninguna raza del
        # dataset. Esto evita clasificaciones forzadas: por ejemplo, si alguien
        # sube la foto de un gato, el modelo va a encontrar "el perro mas
        # parecido" aunque no se parezca en nada; el threshold previene esa
        # falsa clasificacion retornando "unknown".
        if best_score < self.similarity_threshold:
            return ("unknown", best_score)

        # ------------------------------------------------------------------
        # Voto ponderado por score
        # ------------------------------------------------------------------
        # Alternativa: voto simple (cada vecino = 1 voto). El problema es que
        # un vecino lejano (score bajo) pesa igual que uno cercano (score alto).
        #
        # Solucion: voto ponderado. Cada vecino aporta su score como peso.
        # Si hay 3 vecinos de raza "Beagle" con scores 0.9, 0.8, 0.3, el
        # acumulado es 2.0. Si hay 2 de "Pug" con 0.85 y 0.82, el acumulado
        # es 1.67. Gana "Beagle". Esto da mas influencia a los vecinos mas
        # cercanos y confiables.
        votes: dict[str, float] = {}
        for n in results:
            votes[n.breed] = votes.get(n.breed, 0.0) + n.score

        # max(..., key=votes.get) encuentra la raza con mayor acumulado
        predicted = max(votes, key=votes.get)

        # Devolvemos best_score (no el acumulado) como metrica de confianza
        # porque refleja que tan similar es el vecino individual mas cercano,
        # que es la metrica mas interpretable para el usuario.
        return (predicted, round(best_score, 4))

    # ------------------------------------------------------------------
    # Helpers de similitud provistos
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def index_image(
        self, image_path: str, breed: str, metadata: dict[str, object] | None = None
    ) -> EmbeddingRecord:
        """Extrae el embedding de una imagen del dataset y lo persiste en la base vectorial."""
        image = self._load_image(image_path)
        embedding = self.extract_embedding(image)
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(image_path),
            breed=breed,
            metadata=metadata or {},
        )
        self.store.append(record)
        return record

    def _with_url(self, neighbor: Neighbor) -> Neighbor:
        if self.url_resolver is not None and not neighbor.url:
            neighbor.url = self.url_resolver(Path(neighbor.path))
        return neighbor

    def search(
        self,
        source_path: str,
        output_path: Path,
        embedding_fn: Optional[Callable[[np.ndarray], list[float]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Pipeline completo de la Etapa 1: embedding -> vecinos -> raza predicha.

        `embedding_fn` permite seleccionar dinamicamente el extractor
        (baseline, resnet18_finetuned o cnn_custom, ver Etapa 2).
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        extractor = embedding_fn or self.extract_embedding
        embedding = extractor(image)

        k = int(top_k) if top_k else self.top_k
        neighbors = [self._with_url(n) for n in self.search_similar_images(embedding, k)]
        breed, score = self.predict_breed_from_neighbors(neighbors)
        logger.info("Predicted breed: %s (score=%.4f) for %s", breed, score, source_path)

        payload = SearchResult(
            source_path=source_path,
            model=model_name or self.model_name,
            predicted_breed=breed,
            score=round(float(score), 4),
            neighbors=neighbors,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
