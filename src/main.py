"""Simple entry point for the sentiment analysis project."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_model
from src.predict import run_prediction_loop
from src.train import train_model


def show_menu() -> None:
    """Display a small command-line menu."""

    print("Advanced Sentiment Analysis (RNN)")
    print("1. Train model")
    print("2. Evaluate model")
    print("3. Predict sentiment")
    print("4. Exit")


def main() -> None:
    """Run the project from a simple terminal menu."""

    while True:
        show_menu()
        choice = input("Select an option: ").strip()

        if choice == "1":
            train_model()
        elif choice == "2":
            evaluate_model()
        elif choice == "3":
            run_prediction_loop()
        elif choice == "4":
            print("Exiting project.")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
