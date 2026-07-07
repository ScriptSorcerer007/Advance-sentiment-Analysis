"""Run sentiment predictions for custom reviews from the terminal."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import decode_review  # noqa: E402


def load_word_index() -> dict:
    """Load the IMDB word index used by the dataset."""

    return imdb.get_word_index()


def encode_review(review_text: str, word_index: dict, max_words: int = 10_000) -> List[int]:
    """Convert a custom review into a sequence of integer tokens."""

    review_tokens = review_text.lower().split()
    encoded_tokens: List[int] = []

    for token in review_tokens:
        index = word_index.get(token)
        if index is not None and index < max_words:
            encoded_tokens.append(index + 3)
        else:
            encoded_tokens.append(2)

    return encoded_tokens


def predict_sentiment(review_text: str, model, word_index: dict, max_length: int = 200) -> tuple[str, float]:
    """Predict sentiment and confidence for one review."""

    encoded_review = encode_review(review_text, word_index)
    padded_review = pad_sequences([encoded_review], maxlen=max_length, padding="post", truncating="post")
    probability = float(model.predict(padded_review, verbose=0)[0][0])

    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability
    else:
        sentiment = "Negative"
        confidence = 1.0 - probability

    return sentiment, confidence


def run_prediction_loop() -> None:
    """Ask the user for reviews and display predictions."""

    model_path = PROJECT_ROOT / "models" / "best_bilstm_model.keras"
    output_path = PROJECT_ROOT / "outputs" / "predictions" / "custom_predictions.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Model not found at: {model_path}")
        print("Train the model first by running the training script.")
        return

    model = load_model(model_path)
    word_index = load_word_index()

    print("Enter a review to analyze sentiment.")
    print("Type 'exit' to stop.")

    with output_path.open("w", encoding="utf-8") as file:
        while True:
            review_text = input("\nReview: ").strip()
            if review_text.lower() == "exit":
                print("Prediction session ended.")
                break

            if not review_text:
                print("Please enter a non-empty review.")
                continue

            sentiment, confidence = predict_sentiment(review_text, model, word_index)
            encoded_preview = encode_review(review_text, word_index)

            print(f"Predicted Sentiment: {sentiment}")
            print(f"Confidence Score: {confidence:.4f}")

            file.write(f"Review: {review_text}\n")
            file.write(f"Predicted Sentiment: {sentiment}\n")
            file.write(f"Confidence Score: {confidence:.4f}\n")
            file.write(f"Encoded Review Preview: {encoded_preview[:20]}\n")
            file.write("-" * 50 + "\n")


if __name__ == "__main__":
    run_prediction_loop()
