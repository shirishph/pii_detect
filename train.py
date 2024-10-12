import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("train-dataset.tsv", sep='\t')
print(df.head(5))

# Get the Unique Items from the label column
print("Unique labels: ", df['labels'].unique())

# Encode label for easy identification.
df['encoded_cat'] = df['labels'].astype('category').cat.codes
print(df.head())

data_texts = df['data'].to_list() # Features (not tokenized yet)
data_labels = df['encoded_cat'].to_list() # Labels

train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=0.2, random_state=0, shuffle=True)
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.01, random_state=0, shuffle=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=15)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
 
model.fit(
    train_dataset.shuffle(1000).batch(16),
    epochs=2,
    batch_size=16,
    validation_data=val_dataset.shuffle(1000).batch(16),
    callbacks=[early_stopping])

model.summary()

save_directory = "pii_detect_model"

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
