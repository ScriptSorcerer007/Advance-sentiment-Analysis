"""Evaluate the trained Bi-LSTM model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import preprocess_data  # noqa: E402


def load_training_history(history_path: Path) -> dict:
    """Load the saved training history if it exists."""

    if not history_path.exists():
        return {}

    with history_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_metrics_report(metrics: dict, output_path: Path) -> None:
    """Save evaluation metrics to a JSON file."""

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def plot_training_curves(history: dict, graphs_dir: Path) -> None:
    """Plot training and validation accuracy/loss curves."""

    if not history:
        print("Training history not found. Skipping plots.")
        return

    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphs_dir / "accuracy_curve.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphs_dir / "loss_curve.png")
    plt.close()


def evaluate_model() -> None:
    """Load the trained model, evaluate it, and save results."""

    data = preprocess_data()

    models_dir = PROJECT_ROOT / "models"
    outputs_dir = PROJECT_ROOT / "outputs"
    graphs_dir = outputs_dir / "graphs"
    predictions_dir = outputs_dir / "predictions"

    graphs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_bilstm_model.keras"
    history_path = graphs_dir / "training_history.json"

    model = load_model(model_path)
    predictions = model.predict(data.x_test, verbose=0)
    predicted_labels = (predictions >= 0.5).astype(int).reshape(-1)

    accuracy = accuracy_score(data.y_test, predicted_labels)
    precision = precision_score(data.y_test, predicted_labels)
    recall = recall_score(data.y_test, predicted_labels)
    f1 = f1_score(data.y_test, predicted_labels)
    matrix = confusion_matrix(data.y_test, predicted_labels)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
    }

    save_metrics_report(metrics, predictions_dir / "evaluation_metrics.json")

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(graphs_dir / "confusion_matrix.png")
    plt.close()

    history = load_training_history(history_path)
    plot_training_curves(history, graphs_dir)

    print("Evaluation completed successfully")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{matrix}")


if __name__ == "__main__":
    evaluate_model()
