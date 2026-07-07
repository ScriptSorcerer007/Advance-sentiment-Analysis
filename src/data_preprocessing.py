"""Data preprocessing utilities for IMDB sentiment analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


@dataclass
class PreprocessedData:
    """Container for the processed dataset."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    word_index: Dict[str, int]
    max_words: int
    max_length: int


def load_imdb_dataset(max_words: int = 10_000) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Dict[str, int]]:
    """Load the IMDB dataset and its word index.

    Parameters
    ----------
    max_words:
        Keep only the most frequent words.

    Returns
    -------
    ((x_train, y_train), (x_test, y_test), word_index)
    """

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
    word_index = imdb.get_word_index()
    return (x_train, y_train), (x_test, y_test), word_index


def inspect_dataset(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """Print a small dataset summary for learning purposes."""

    print("Dataset summary")
    print(f"Training samples: {len(x_train)}")
    print(f"Training labels: {len(y_train)}")
    print(f"First review length: {len(x_train[0]) if len(x_train) > 0 else 0}")
    print(f"Sample label distribution: positive={int(y_train.sum())}, negative={len(y_train) - int(y_train.sum())}")


def handle_missing_values(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove empty reviews or missing labels if they appear.

    The IMDB dataset normally does not contain missing values, but this
    function keeps the pipeline safe and easy to explain.
    """

    cleaned_reviews = []
    cleaned_labels = []

    for review, label in zip(x_data, y_data):
        if review is None:
            continue
        if len(review) == 0:
            continue
        if label is None:
            continue
        cleaned_reviews.append(review)
        cleaned_labels.append(label)

    return np.array(cleaned_reviews, dtype=object), np.array(cleaned_labels)


def pad_review_sequences(
    x_train: np.ndarray,
    x_test: np.ndarray,
    max_length: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad or truncate all reviews to the same length."""

    x_train_padded = pad_sequences(x_train, maxlen=max_length, padding="post", truncating="post")
    x_test_padded = pad_sequences(x_test, maxlen=max_length, padding="post", truncating="post")
    return x_train_padded, x_test_padded


def preprocess_data(max_words: int = 10_000, max_length: int = 200) -> PreprocessedData:
    """Load, inspect, clean, and pad the IMDB dataset."""

    (x_train, y_train), (x_test, y_test), word_index = load_imdb_dataset(max_words=max_words)

    inspect_dataset(x_train, y_train)

    x_train, y_train = handle_missing_values(x_train, y_train)
    x_test, y_test = handle_missing_values(x_test, y_test)

    x_train_padded, x_test_padded = pad_review_sequences(x_train, x_test, max_length=max_length)

    return PreprocessedData(
        x_train=x_train_padded,
        x_test=x_test_padded,
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        word_index=word_index,
        max_words=max_words,
        max_length=max_length,
    )


def decode_review(encoded_review: np.ndarray, word_index: Dict[str, int]) -> str:
    """Convert a tokenized review back into readable text for inspection."""

    reverse_word_index = {index + 3: word for word, index in word_index.items()}
    reverse_word_index[0] = "<PAD>"
    reverse_word_index[1] = "<START>"
    reverse_word_index[2] = "<UNK>"
    reverse_word_index[3] = "<UNUSED>"

    return " ".join(reverse_word_index.get(int(token), "<UNK>") for token in encoded_review)


if __name__ == "__main__":
    data = preprocess_data()
    print("Preprocessing complete")
    print(f"Train shape: {data.x_train.shape}")
    print(f"Test shape: {data.x_test.shape}")
