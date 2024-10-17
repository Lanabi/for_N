import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
import pandas as pd
import numpy
import time
import pickle
import os 

# Get the current working directory
current_directory = os.getcwd()
# Construct the path to open the file file
for_rubert = os.path.join(current_directory, "for_rubert.pkl")
# Construct the path to save the output file 
from_rubert_path = os.path.join(current_directory, "from_rubert.pkl")

# Open the file
df = pd.read_pickle(for_rubert) 
texts = df['text']

# Labelling
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

def get_sentiment(text, return_type='label'):
    """ Calculate sentiment of a text. `return_type` can be 'label', 'score' or 'proba' """
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
    if return_type == 'label':
        return model.config.id2label[proba.argmax()]
    elif return_type == 'score':
        return proba.dot([-1, 0, 1])
    return proba

# Classify the text
labels = [get_sentiment(i, 'label') for i in texts]

# Score the text on the scale from -1 (very negative) to +1 (very positive)
score = [get_sentiment(i, 'score') for i in texts]
# calculate probabilities of all labels
predictions1 = [get_sentiment(i, 'proba')[0] for i in texts]
predictions2 = [get_sentiment(i, 'proba')[1] for i in texts]
predictions3 = [get_sentiment(i, 'proba')[2] for i in texts]

df['labels'] = labels
df['score'] = score

# Save the labelled dataframe
df.to_pickle(from_rubert_path) 
