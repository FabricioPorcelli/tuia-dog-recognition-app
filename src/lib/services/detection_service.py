from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import onnxruntime
import torch
from torchvision import datasets, transforms
from ultralytics import YOLO

from lib.config import IMAGENET_MEAN, IMAGENET_STD
from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService

logger = logging.getLogger(__name__)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones a implementar por el estudiante:
      - detect_dogs(image)
      - classify_detected_dog(crop)

    La orquestacion (predict: deteccion -> recorte -> clasificacion -> JSON)
    ya esta provista.
    """

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV / ultralytics)
        return image

    # ------------------------------------------------------------------
    # Etapa 3: funciones a implementar
    # ------------------------------------------------------------------

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Detecta todos los perros presentes en la imagen usando un modelo YOLO
        pre-entrenado (ej: YOLOv8n via ultralytics). No es necesario entrenar
        el detector.

        Sugerencias:
          - self.yolo_model_name, self.conf_threshold y self.dog_class_id
            (clase 'dog' = 16 en COCO) vienen de la configuracion (.env).
          - Debe funcionar con un perro, multiples perros y escenas complejas.

        Retorna una lista de ((x1, y1, x2, y2), confidence) en pixeles.
        """

        # Cache del modelo YOLO para evitar recargar el .pt en cada llamada.
        # Ultralytics descarga automaticamente el modelo la primera vez.
        if not hasattr(self, '_yolo_model'):
            self._yolo_model = YOLO(self.yolo_model_name)

        # Inferencia sobre la imagen original (BGR). YOLO maneja la conversion
        # interna a RGB. verbose=False silencia los logs de ultralytics.
        results = self._yolo_model(image, verbose=False)[0]

        # Recorremos todas las detecciones y filtramos solo perros (COCO id 16)
        # que superen el umbral de confianza configurado.
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            if cls_id == self.dog_class_id and conf >= self.conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(((int(x1), int(y1), int(x2), int(y2)), conf))

        return detections

    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza del recorte de un perro detectado usando el modelo
        entrenado en la Etapa 2 (self.classifier.load_model()).

        El recorte llega en BGR (OpenCV). Retorna (raza, score).
        """
        # Cargamos el modelo entrenado en Etapa 2 (con cache interno del
        # ClassifierService). Soportamos .pth (PyTorch) y .onnx (ONNX).
        model_raw = self.classifier.load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_onnx = isinstance(model_raw, onnxruntime.InferenceSession)

        # Pipeline de preprocesamiento identico al usado en entrenamiento
        # (Resize + ToTensor + Normalize con medias/std de ImageNet).
        # No aplicamos data augmentation porque es inferencia.
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.classifier.image_size, self.classifier.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # Convertimos de BGR (OpenCV) a RGB y anyadimos dimension batch.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb).unsqueeze(0)

        # Cache de nombres de razas: leemos la estructura de directorios
        # de validacion una sola vez. ImageFolder asigna los mismos indices
        # que durante entrenamiento (orden alfabetico por nombre de carpeta).
        if not hasattr(self, '_class_names'):
            val_dataset = datasets.ImageFolder(
                root=str(self.classifier.dataset_path / "valid"),
                transform=transforms.Compose([
                    transforms.Resize((self.classifier.image_size, self.classifier.image_size)),
                    transforms.ToTensor(),
                ]),
            )
            self._class_names = val_dataset.classes

        # Inferencia con el formato del checkpoint.
        # ONNX requiere entrada numpy y no soporta .to(device).
        # PyTorch usa torch.no_grad() para eficiencia.
        if is_onnx:
            input_name = model_raw.get_inputs()[0].name
            outputs = model_raw.run(None, {input_name: input_tensor.cpu().numpy()})[0]
            probs = torch.softmax(torch.from_numpy(outputs), dim=1)
        else:
            model = model_raw.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(input_tensor.to(device))
                probs = torch.softmax(outputs, dim=1)

        # Tomamos la clase con mayor probabilidad y la mapeamos a nombre de raza.
        pred_idx = torch.argmax(probs, dim=1).item()
        score = probs[0, pred_idx].item()
        breed = self._class_names[pred_idx]

        return (breed, score)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        """Clasifica la imagen completa con el modelo entrenado (pestaña Etapa 2).

        Reutiliza classify_detected_dog tratando la imagen entera como recorte,
        por lo que requiere la Etapa 2 (modelo entrenado) y classify_detected_dog.
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        if model_name:
            self.classifier.set_active_model(model_name)
        breed, score = self.classify_detected_dog(image)
        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        """Flujo completo: deteccion -> bounding boxes -> recortes -> clasificacion.

        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score = self.classify_detected_dog(crop)
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
