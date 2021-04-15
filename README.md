# Duplicate-Finder
A Siamese network model trained on Quora question dataset which detects duplicates with an Accuracy and F1 score of ~80%. 

Trained using [TRAX](https://github.com/google/trax): An end-to-end library for Deep Learning that focuses on clear code and speed. It is actively used and maintained in the Google Brain team.

## Dataset
The dataset consists of more than 400k question pairs from Quora, where each pair is labeled if the pair is duplicate or not. As per the need of this model only the duplicate pairs are used to train the model, which consists of around 150k pairs.

Dataset is hosted [here](https://www.kaggle.com/c/quora-question-pairs/data) on Kaggle.

## Model Overview
A parallel model architecture is used for combining two similar metworks for each questions. These networks consists of an Embedding layer and multiple LSTM layers. The normalized mean vector from each model is used to calculate loss and train the model.

### Loss Function
We used triplet loss to train the model. Only the positive (duplicate) pairs are used from the dataset. Negative (non-duplicate) pairs are achieved by pairing one sentence with another in the same batch. Then the most similar sentence pairs from non-duplicate was weighted more to train a stronger model. By trial and error, we found a margin of 0.5 (difference between positive and negative pair's cosine similarity) to be suitable for an unbiased model training.

## Results
Model was tested on 10000 unseen data. With a threshold of 0.7 our model achieves an accuracy of 79.44% and an F1 score of 81.26%.
