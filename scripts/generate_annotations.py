"""Genera anotaciones automaticas (YOLOv5 o COCO) para una carpeta de imagenes (Etapa 4).

Requiere haber implementado PipelineService.generate_annotations (y las
funciones de la Etapa 3 que este reutiliza).

Uso:
    python scripts/generate_annotations.py <carpeta_de_imagenes> [--format yolo|coco]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Directorio desde donde se invoco el script (para resolver rutas pasadas por CLI).
LAUNCH_DIR = Path.cwd()

# Usa la misma configuracion que el backend local (src/.env, con paths relativos a src/).
# Si src/.env no existe, se usan los defaults con paths relativos a la raiz del repo.
os.chdir(SRC if (SRC / ".env").is_file() else ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Carpeta con imagenes a procesar.")
    parser.add_argument(
        "--format",
        dest="output_format",
        default="yolo",
        choices=("yolo", "coco"),
        help="Formato de salida de las anotaciones.",
    )
    args = parser.parse_args()

    from lib.bootstrap import build_classifier, build_detection, build_pipeline
    from lib.config import settings

    classifier = build_classifier(settings)
    detection = build_detection(settings, classifier)
    pipeline = build_pipeline(settings, detection)

    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = (LAUNCH_DIR / folder).resolve()

    output = pipeline.generate_annotations(str(folder), args.output_format)
    print(f"Anotaciones generadas en: {output}")


if __name__ == "__main__":
    main()
