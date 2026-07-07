"""Model definition for sentiment analysis using Bi-LSTM."""

from __future__ import annotations

from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_bilstm_model(
    max_words: int = 10_000,
    embedding_dim: int = 128,
    max_length: int = 200,
) -> Sequential:
    """Build and compile a Bidirectional LSTM model for binary sentiment classification."""

    model = Sequential(
        [
            Embedding(
                input_dim=max_words,
                output_dim=embedding_dim,
                input_length=max_length,
                name="embedding_layer",
            ),
            Bidirectional(
                LSTM(64, return_sequences=False),
                name="bidirectional_lstm",
            ),
            Dropout(0.5, name="dropout_layer"),
            Dense(32, activation="relu", name="dense_layer"),
            Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="bi_lstm_sentiment_model",
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    sentiment_model = build_bilstm_model()
    sentiment_model.summary()
