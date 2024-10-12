from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

save_directory = "pii_detect_model"

loaded_tokenizer = DistilBertTokenizer.from_pretrained(save_directory)
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

df = pd.read_csv("holdout-dataset.tsv", sep='\t')
print(df.head(5))

# Get the Unique Items from the label column
print("Unique labels: ", df['labels'].unique())

# Encode label for easy identification.
df['encoded_cat'] = df['labels'].astype('category').cat.codes
print(df.head())

data_texts = df['data'].to_list() # Features (not tokenized yet)
data_labels = df['encoded_cat'].to_list() # Labels

y_true = []
y_pred = []
for text, label in zip(data_texts, data_labels):
    print("> ", str(text) + "  |  " + str(label))

    test_text = text
    y_true.append(label)

    # Business = 0, Entertainment = 1, Politics = 2, Sport = 3, Tech = 4
    predict_input = loaded_tokenizer.encode(test_text,
        truncation=True,
        padding=True,
        return_tensors="tf")
 
    output = loaded_model(predict_input)[0]
 
    prediction_value = tf.argmax(output, axis=1).numpy()[0]
    print("prediction_value: ", prediction_value)

    predict_input = loaded_tokenizer.encode(test_text,
        truncation=True,
        padding=True,
        return_tensors="tf")

    output = loaded_model(predict_input)[0]

    prediction_value = tf.argmax(output, axis=1).numpy()[0]
    y_pred.append(prediction_value)

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

print("y_true: ", y_true)
print("y_pred: ", y_pred)

print(classification_report(y_true, y_pred))
