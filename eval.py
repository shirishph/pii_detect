from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report

from tensorflow.keras.models import load_model
save_directory = "Multitext_Classification_colab" # Change this to your preferred location

loaded_tokenizer = DistilBertTokenizer.from_pretrained(save_directory)
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

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
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.05, random_state=0, shuffle=True)

test_text = data_texts[1842]
print("test_text: ", test_text)

y_true = []
y_pred = []
for i in range(len(test_texts)):
    print("* test_texts: ", test_texts[i][:50], " --- test_labels: ", test_labels[i])

    test_text = test_texts[i]
    y_true.append(test_labels[i])

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

    # Convert numeric prediction to category label
    if prediction_value == 0:
        prediction_label = "Business"
    elif prediction_value == 1:
        prediction_label = "Entertainment"
    elif prediction_value == 2:
        prediction_label = "Politics"
    elif prediction_value == 3:
        prediction_label = "Sport"
    else:
        prediction_label = "Tech" # Handle unexpected values if necessary

    print("🤖Predicted Category:", prediction_label)

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

print("y_true: ", y_true)
print("y_pred: ", y_pred)

print(classification_report(y_true, y_pred))

