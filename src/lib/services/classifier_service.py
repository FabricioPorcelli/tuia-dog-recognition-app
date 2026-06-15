from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import onnxruntime
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from lib.config import IMAGENET_MEAN, IMAGENET_STD
from lib.evaluation.metrics import precision_recall_f1, specificity

logger = logging.getLogger(__name__)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """Entrena el clasificador de razas sobre self.dataset_path.

        Modelo A (obligatorio): fine-tuning de ResNet18 pre-entrenado en ImageNet.
        Modelo B (opcional): CNN propia desde cero.
        Guarda el checkpoint en self.active_checkpoint.
        """
        # ------------------------------------------------------------------
        # Hiperparametros
        # ------------------------------------------------------------------
        # num_epochs=10: punto de partida suficiente para convergencia con
        # fine-tuning. Para la CNN custom pueden requerirse mas epochs.
        # batch_size=32: balance entre memoria GPU y estabilidad del gradiente.
        # learning_rate=0.001: valor tipico para Adam, funciona bien en
        # fine-tuning sin necesidad de calentamiento.
        num_epochs = 10
        batch_size = 32
        learning_rate = 0.001
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Usando dispositivo: %s", device)

        # ------------------------------------------------------------------
        # Transformaciones y data augmentation
        # ------------------------------------------------------------------
        # Para training aplicamos data augmentation (RandomHorizontalFlip,
        # ColorJitter) que reduce overfitting al exponer el modelo a
        # variaciones de orientacion e iluminacion. En validacion solo
        # preprocesamos (sin augmentation) para medir rendimiento real.
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # ------------------------------------------------------------------
        # Datasets y DataLoaders
        # ------------------------------------------------------------------
        # ImageFolder asume estructura dataset/{train,valid}/{raza}/*.jpg.
        # Asigna automaticamente un indice numerico a cada raza.
        # shuffle=True en train evita que el modelo aprenda el orden de las
        # muestras; en validacion no hace falta porque solo evaluamos.
        train_dataset = datasets.ImageFolder(
            root=str(self.dataset_path / "train"),
            transform=train_transform,
        )
        val_dataset = datasets.ImageFolder(
            root=str(self.dataset_path / "valid"),
            transform=val_transform,
        )

        num_classes = len(train_dataset.classes)
        logger.info(
            "Clases: %d | Train: %d imagenes | Val: %d imagenes",
            num_classes, len(train_dataset), len(val_dataset),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        )

        # ------------------------------------------------------------------
        # Construccion del modelo segun self.active_model_name
        # ------------------------------------------------------------------
        # Modelo A (resnet18_finetuned):
        #   ResNet18 pre-entrenado en ImageNet. Reemplazamos la capa fc
        #   (original: 512->1000 clases) por una nueva 512->num_classes.
        #   El backbone ya reconoce bordes, texturas y formas gracias a
        #   ImageNet; solo reentrenamos la ultima capa (fine-tuning).
        #
        # Modelo B (cnn_custom):
        #   CNN desde cero con 5 bloques Conv2D + BatchNorm + ReLU + MaxPool,
        #   seguidas de AdaptiveAvgPool2d, Flatten y una capa Linear final.
        #   Arquitectura simple pero suficiente para el dataset.
        if self.active_model_name == "resnet18_finetuned":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            classifier = nn.Linear(512, num_classes)
            model = nn.Sequential(backbone, classifier)

        model = model.to(device)

        # ------------------------------------------------------------------
        # Funcion de perdida y optimizador
        # ------------------------------------------------------------------
        # CrossEntropyLoss: funcion estandar para clasificacion multiclase.
        # Combina LogSoftmax + NLLLoss. Adam: optimizador adaptativo que
        # ajusta la tasa de aprendizaje por parametro.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # ------------------------------------------------------------------
        # Bucle de entrenamiento con validacion por epoch
        # ------------------------------------------------------------------
        # Por cada epoch: entrenamos sobre train_loader (con backward) y
        # luego evaluamos sobre val_loader (sin gradientes) para monitorear
        # overfitting. La validacion usa torch.no_grad() por eficiencia.
        for epoch in range(num_epochs):
            # --- Train ---
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix(loss=loss.item(), acc=correct / total)

            train_acc = 100.0 * correct / total
            logger.info("Epoch %d: Train Acc: %.2f%%", epoch + 1, train_acc)

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = 100.0 * correct / total
            logger.info("Epoch %d: Val Acc: %.2f%%\n", epoch + 1, val_acc)

        # ------------------------------------------------------------------
        # Guardar checkpoint
        # ------------------------------------------------------------------
        # torch.save(model, path) persiste el modelo completo (arquitectura
        # + pesos) en formato pickle, permitiendo cargarlo con torch.load()
        # sin redefinir la clase. El directorio se crea si no existe.
        self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, self.active_checkpoint)
        logger.info("Checkpoint guardado en: %s", self.active_checkpoint)

    def evaluate_classifier(self) -> dict[str, float]:
        """Evalua el modelo activo sobre el conjunto de prueba (test split).

        Reporta accuracy, precision, recall, specificity y F1-Score usando
        macro-promedio. Retorna dict con las 5 metricas redondeadas a 4 decimales.
        """
        # ------------------------------------------------------------------
        # Cargar modelo y determinar backend (PyTorch vs ONNX)
        # ------------------------------------------------------------------
        # Soportamos ambos formatos porque la notebook de Colab puede exportar
        # a .onnx. ONNX requiere un flujo de inferencia distinto (numpy en
        # vez de tensores de torch).
        model_raw = self.load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_onnx = isinstance(model_raw, onnxruntime.InferenceSession)

        # ------------------------------------------------------------------
        # Preprocesamiento para evaluacion (sin data augmentation)
        # ------------------------------------------------------------------
        # Usamos el mismo pipeline que en similarity_service para consistencia:
        # resize a 224x224, conversion a tensor y normalizacion ImageNet.
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # ------------------------------------------------------------------
        # Dataset de prueba
        # ------------------------------------------------------------------
        # Test split se mantiene separado desde el dataset original de Kaggle.
        # Son imagenes que el modelo jamas vio durante entrenamiento ni
        # validacion, dando una estimacion realista del rendimiento.
        test_dataset = datasets.ImageFolder(
            root=str(self.dataset_path / "test"),
            transform=transform,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=2,
        )

        # ------------------------------------------------------------------
        # Inferencia sobre todo el test set
        # ------------------------------------------------------------------
        # Recolectamos predicciones y etiquetas reales para calcular metricas.
        # El manejo es distinto segun el formato del modelo.
        all_preds: list[int] = []
        all_labels: list[int] = []

        if is_onnx:
            # ONNX: sesion de inferencia con entradas/salidas fijas.
            # No soporta autograd ni .to(device); la entrada es numpy.
            input_name = model_raw.get_inputs()[0].name
            for inputs, labels in test_loader:
                inputs_np = inputs.cpu().numpy()
                outputs = model_raw.run(None, {input_name: inputs_np})[0]
                preds = np.argmax(outputs, axis=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        else:
            # PyTorch: forward pass estandar con torch.no_grad() para
            # eficiencia (no necesitamos gradientes en evaluacion).
            model = model_raw.to(device)
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

        # ------------------------------------------------------------------
        # Calculo de metricas desde la matriz de confusion
        # ------------------------------------------------------------------
        # Construimos la matriz de confusion: cm[i,j] = imagenes de clase i
        # predichas como clase j. A partir de ella obtenemos TP, FP, FN, TN
        # por cada clase y calculamos las 5 metricas usando las helpers de
        # lib.evaluation.metrics.
        #
        # Usamos macro-promedio (promediar metricas de cada clase) en lugar
        # de micro-promedio porque da igual peso a todas las razas,
        # independientemente de su frecuencia en el dataset. Esto evita que
        # razas mayoritarias dominen las metricas.
        cm = np.zeros((len(test_dataset.classes), len(test_dataset.classes)), dtype=int)
        for true_label, pred_label in zip(all_labels, all_preds):
            cm[true_label, pred_label] += 1

        num_classes = cm.shape[0]
        total = cm.sum()
        correctos = int(np.trace(cm))
        accuracy = correctos / total if total > 0 else 0.0

        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        specificities_list: list[float] = []

        for i in range(num_classes):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            tn = int(total - tp - fp - fn)

            p, r, f = precision_recall_f1(tp, fp, fn)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            specificities_list.append(specificity(tn, fp))

        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))
        f1 = float(np.mean(f1s))
        specificity_val = float(np.mean(specificities_list))

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity_val, 4),
            "f1": round(f1, 4),
        }

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """Genera el embedding usando el modelo propio activo (penultima capa).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados en Etapa 2.
        La imagen llega en BGR (OpenCV). Retorna lista de floats.
        """
        # ------------------------------------------------------------------
        # Cargar modelo entrenado (una sola vez, con cache)
        # ------------------------------------------------------------------
        # Usamos hasattr para cachear el modelo como atributo de instancia,
        # evitando recargar el checkpoint de disco en cada llamada.
        if not hasattr(self, '_custom_emb_model'):
            model_raw = self.load_model()

            # ONNX no permite remover capas dinamicamente porque su grafo
            # de computo es estatico. Solo los checkpoints .pth (PyTorch)
            # soportan extraer la penultima capa con children()[:-1].
            if isinstance(model_raw, onnxruntime.InferenceSession):
                raise ValueError(
                    "extract_custom_embedding no soporta formato ONNX. "
                    "Usa un checkpoint .pth."
                )

            # El checkpoint guarda el modelo completo con su capa de
            # clasificacion al final. Nosotros necesitamos el vector de
            # features ANTES de esa capa (el embedding). Con children()[:-1]
            # descartamos el ultimo modulo y nos quedamos solo con el backbone.
            self._custom_emb_model = torch.nn.Sequential(
                *list(model_raw.children())[:-1]
            )
            self._custom_emb_model.eval()
            self._custom_emb_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._custom_emb_model = self._custom_emb_model.to(
                self._custom_emb_device
            )

        # ------------------------------------------------------------------
        # Pipeline de preprocesamiento (identico a similarity_service)
        # ------------------------------------------------------------------
        # Mismo pipeline que extract_embedding en similarity_service para que
        # los embeddings generados por ambos extractores sean comparables
        # bajo la misma metrica de similitud.
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        # ------------------------------------------------------------------
        # BGR -> RGB, preprocesar, forward
        # ------------------------------------------------------------------
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb).unsqueeze(0).to(self._custom_emb_device)

        with torch.no_grad():
            emb = self._custom_emb_model(input_tensor).squeeze().cpu().numpy()

        return emb.tolist()
