# pii_detect
This project aims to identify Personally Identifiable Information (PII) within datasets using machine learning techniques.

It involves three primary stages:
* **Dataset Generation**: A synthetic dataset is created, containing commonly encountered PII such as names, addresses, phone numbers, and social security numbers.
* **Model Development**: A machine learning model is designed and trained on this synthetic dataset to accurately detect and classify PII data.
* **Evaluation**: The trained model is evaluated to measure its performance in identifying PII, using standard metrics like accuracy, precision, recall, and F1 score.

This repository is ideal for developers working in data privacy, compliance, and security, offering a foundation for building systems to automatically detect sensitive information in structured or unstructured datasets.

There are 15 detectable PII entities but more can be added.

**Steps**
```sh
$ git clone https://github.com/shirishph/pii_detect.git
$ cd pii_detect
$ virtualenv venv -p python3.10
$ source ./venv/bin/activate
$ pip install -r requirements.txt
# Adjust TOTAL_TRAINING_DATAPOINTS and TOTAL_HOLDOUT_DATAPOINTS in file below
$ python3 gen.py
$ wc -l *.tsv
    1001 holdout-dataset.tsv
   20001 train-dataset.tsv
$ python3 train.py
$ python3 eval.py
```
**Classification Report** 
```sh
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        67
           1       1.00      1.00      1.00        67
           2       1.00      1.00      1.00        67
           3       1.00      1.00      1.00        67
           4       1.00      1.00      1.00        67
           5       1.00      1.00      1.00        67
           6       1.00      1.00      1.00        67
           7       1.00      1.00      1.00        67
           8       1.00      0.97      0.98        67
           9       1.00      1.00      1.00        67
          10       0.94      1.00      0.97        66
          11       1.00      1.00      1.00        66
          12       1.00      1.00      1.00        66
          13       0.97      0.94      0.95        66
          14       1.00      1.00      1.00        66

    accuracy                           0.99      1000
   macro avg       0.99      0.99      0.99      1000
weighted avg       0.99      0.99      0.99      1000
```

Loosely based on this implementation:
https://medium.com/@kiddojazz/distilbert-for-multiclass-text-classification-using-transformers-d6374e6678ba

