from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

# Model name
MODEL_NAME = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

class SentimentAnalyzer:
    def __init__(self):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.labels = ['Negative', 'Neutral', 'Positive']
        print("Model loaded.")

    def analyze(self, text):
        """
        Analyzes the sentiment of the text.
        
        Args:
            text (str): The input text.
            
        Returns:
            dict: A dictionary containing the label and confidence score.
        """
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        
        top_label = self.labels[ranking[0]]
        top_score = scores[ranking[0]]
        
        return {
            "label": top_label,
            "score": float(top_score),
            "all_scores": {self.labels[i]: float(scores[i]) for i in range(len(scores))}
        }
