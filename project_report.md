# Project Report: Advanced Sentiment Analysis (RNN)

## Abstract

This project builds a simple deep learning system for sentiment analysis using the IMDB movie reviews dataset. The goal is to classify each review as positive or negative using a Bidirectional LSTM model with word embeddings. The project demonstrates the complete pipeline of text preprocessing, model building, training, evaluation, and prediction in a form that is suitable for an undergraduate academic assignment.

## Introduction

Sentiment analysis is a Natural Language Processing task that identifies the emotional tone of text. It is commonly used for reviews, comments, and feedback analysis. In this project, the model learns from movie reviews and predicts whether a review expresses positive or negative sentiment.

A Bidirectional LSTM is used because it can understand context from both the beginning and the end of a sentence. This is helpful for text data, where the meaning of a word often depends on nearby words.

## Methodology

The project uses the IMDB dataset from TensorFlow/Keras. Reviews are converted into integer token sequences using tokenization. Because reviews have different lengths, padding is applied so that every review has the same input size.

The model architecture includes:

- An Embedding layer to convert token IDs into dense vectors
- A Bidirectional LSTM layer to learn context in both directions
- A Dropout layer to reduce overfitting
- Dense layers to learn the final classification mapping
- A Sigmoid output layer for binary classification

The model is trained using binary cross-entropy loss and the Adam optimizer.

## Evaluation

The model is evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Training and validation accuracy/loss plots are also generated to show how learning progresses over epochs.

## Results

The final trained model can classify unseen reviews entered from the terminal. The prediction script returns the predicted sentiment and a confidence score.

## Conclusion

This project demonstrates how deep learning can be applied to sentiment analysis using an RNN-based architecture. It also shows the importance of preprocessing, tokenization, padding, and embeddings in text classification tasks.

## Future Scope

Possible future improvements include:

- Testing on a larger dataset such as Amazon Product Reviews
- Using stacked LSTM layers
- Comparing Bi-LSTM with GRU and CNN models
- Improving text cleaning and vocabulary handling
- Visualizing word importance or attention-based explanations
