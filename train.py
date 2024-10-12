from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd

# url = 'https://github.com/kiddojazz/Multitext-Classification/blob/master/bbc_data.csv?raw=true'
# df = pd.read_csv(url)
df = pd.read_csv("bbc_data.csv")
print(df.head(5))

# Get the Unique Items from the label column
df['labels'].unique()

# Filter out rows where the labels are ‘unknown’
df = df[df['labels'] != 'unknown']

# Print the filtered DataFrame
print(df.head())

# Encode label for easy identification.
df['encoded_cat'] = df['labels'].astype('category').cat.codes
print(df.head())

filtered_df = df[df['encoded_cat'] == 4]

# Print the filtered DataFrame
print(filtered_df)

data_texts = df['data'].to_list() # Features (not tokenized yet)
data_labels = df['encoded_cat'].to_list() # Labels

from sklearn.model_selection import train_test_split

# Split Train and Validation data
train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=0.2, random_state=0, shuffle=True)

# Keep some data for inference (testing)
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.01, random_state=0, shuffle=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])

#from tensorflow.keras.callbacks import EarlyStopping
 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
 
model.fit(
    train_dataset.shuffle(1000).batch(16),
    epochs=2,
    batch_size=16,
    validation_data=val_dataset.shuffle(1000).batch(16),
    callbacks=[early_stopping])

model.summary()

from tensorflow.keras.models import load_model
save_directory = "Multitext_Classification_colab" # Change this to your preferred location

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)



