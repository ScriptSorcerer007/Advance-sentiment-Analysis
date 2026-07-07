"""Train the Bi-LSTM sentiment analysis model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import preprocess_data  # noqa: E402
from src.model import build_bilstm_model  # noqa: E402


def train_model() -> None:
    """Load data, build the model, train it, and save the best version."""

    data = preprocess_data()
    model = build_bilstm_model(
        max_words=data.max_words,
        max_length=data.max_length,
    )

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    best_model_path = model_dir / "best_bilstm_model.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        data.x_train,
        data.y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = model_dir / "final_bilstm_model.keras"
    model.save(final_model_path)

    graphs_dir = PROJECT_ROOT / "outputs" / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    history_path = graphs_dir / "training_history.json"

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=4)

    print("Training completed successfully")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    train_model()
