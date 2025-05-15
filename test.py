from transformers import pipeline
from transformers import BertForSequenceClassification, BertTokenizer

# Load model 
model = BertForSequenceClassification.from_pretrained("milanmayur20/imdb-genre-bert")
tokenizer = BertTokenizer.from_pretrained("milanmayur20/imdb-genre-bert")

# Use model
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# text
text = "They were laughing at his fall."

# Predict
prediction = classifier(text)
print(prediction)

