# Advanced Sentiment Analysis (RNN)

This project is a simple academic deep learning project for sentiment analysis using a Bidirectional LSTM model.

## Project Goal

The goal is to classify movie reviews from the IMDB dataset as positive or negative using:

- Natural Language Processing
- Tokenization and padding
- Word embeddings
- Bidirectional LSTM
- Model evaluation with common classification metrics

## Dataset

The project will use the IMDB Movie Reviews dataset from TensorFlow/Keras.

## Folder Structure

- `dataset/` - stores dataset-related files if needed
- `models/` - stores trained model files
- `outputs/graphs/` - stores training and evaluation plots
- `outputs/predictions/` - stores prediction results
- `src/` - contains the Python source code

## Planned Files

- `src/data_preprocessing.py` - load and prepare the text data
- `src/model.py` - define the Bi-LSTM model
- `src/train.py` - train the model
- `src/evaluate.py` - evaluate the model
- `src/predict.py` - run custom sentiment predictions
- `src/main.py` - terminal entry point for running the project step by step

## Completed Files

- `project_report.md` - full project report with abstract, introduction, conclusion, and future scope
- `src/main.py` - simple menu to train, evaluate, or predict from the terminal

## Project Report Topics

The report covers:

- Abstract
- Introduction
- Methodology
- Evaluation
- Results
- Conclusion
- Future Scope

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```

If you already have the project virtual environment, activate it first:

```powershell
Set-Location 'D:\RNN based project\Advanced_Sentiment_Analysis'
& '..\.venv\Scripts\Activate.ps1'
```

Then install requirements inside that environment:

```powershell
python -m pip install -r requirements.txt
```

## Next Steps

The codebase now includes the full workflow for preprocessing, model building, training, evaluation, and prediction.

To run the project, start with `src/main.py` after installing the requirements and training the model.

## Run Command

From the project folder, run:

```powershell
python .\src\main.py
```

If `python` does not point to your virtual environment, use the venv Python directly:

```powershell
& '..\.venv\Scripts\python.exe' .\src\main.py
```
