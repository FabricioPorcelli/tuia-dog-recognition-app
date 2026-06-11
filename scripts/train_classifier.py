"""Entrena y evalua el clasificador de razas (Etapa 2).

Requiere haber implementado ClassifierService.train_classifier y
ClassifierService.evaluate_classifier.

Uso:
    python scripts/train_classifier.py [--model resnet18_finetuned|cnn_custom]
                                       [--skip-train] [--skip-eval]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Usa la misma configuracion que el backend local (src/.env, con paths relativos a src/).
# Si src/.env no existe, se usan los defaults con paths relativos a la raiz del repo.
os.chdir(SRC if (SRC / ".env").is_file() else ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="resnet18_finetuned",
        choices=("resnet18_finetuned", "cnn_custom"),
        help="Modelo a entrenar/evaluar.",
    )
    parser.add_argument("--skip-train", action="store_true", help="Solo evaluar.")
    parser.add_argument("--skip-eval", action="store_true", help="Solo entrenar.")
    args = parser.parse_args()

    from lib.bootstrap import build_classifier
    from lib.config import settings

    classifier = build_classifier(settings)
    classifier.set_active_model(args.model)

    if not args.skip_train:
        print(f"Entrenando {args.model} ...")
        classifier.train_classifier()
        print(f"Checkpoint esperado en: {classifier.active_checkpoint}")

    if not args.skip_eval:
        print(f"Evaluando {args.model} ...")
        metrics = classifier.evaluate_classifier()
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
