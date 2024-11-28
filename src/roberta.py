from preprocess import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
from scipy.special import softmax
import numpy as np
from functools import lru_cache
from typing import Dict

class NFLSentimentAnalyzer:
    def __init__(self):
        self.MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)
        self.config = AutoConfig.from_pretrained(self.MODEL)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def preprocess(self, text: str) -> str:
        """RoBERTa Preprocess text for the model"""
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    @lru_cache(maxsize=1024)
    def get_sentiment_scores(self, text: str) -> dict:
        """Get sentiment scores for a single text"""
        try:
            # Preprocess and encode
            text = self.preprocess(text)
            encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Get model output
            with torch.no_grad():
                output = self.model(**encoded_input)
            
            # Process scores
            scores = output[0][0].cpu().numpy()
            scores = softmax(scores)
            
            # Apply temperature scaling to make predictions more extreme
            temperature = 0.7 
            scores = softmax(scores / temperature)
            
            # Redistribute low-confidence neutral predictions
            neutral_threshold = 0.5
            if scores[1] < neutral_threshold:  # If neutral score is low
                # Zero out neutral and renormalize between positive/negative
                scores[1] = 0
                scores = scores / scores.sum()
            
            return {
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2]),
                'label': self.config.id2label[scores.argmax()],
                'score': float(scores.max()),
                'raw_outputs': scores.tolist()
            } 
        except Exception as e:
            print(f"Error in get_sentiment_scores: {str(e)}")
            return {
                'negative': 0.0,
                'neutral': 1.0, 
                'positive': 0.0,
                'label': 'neutral',
                'score': 1.0,
                'raw_outputs': [0.0, 1.0, 0.0]
            }
    
    def batch_analyze(self, texts: list, batch_size: int = 16) -> list:
        """Analyze sentiment for a batch of texts"""
        results = []
        for i in range(0, len(texts), batch_size):
            try:
                batch_texts = texts[i:i + batch_size]
                batch_processed = [self.preprocess(text) for text in batch_texts]
                
                # Tokenize batch
                encoded_input = self.tokenizer(batch_processed, 
                                             return_tensors='pt', 
                                             padding=True, 
                                             truncation=True, 
                                             max_length=128)
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Get model output
                with torch.no_grad():
                    output = self.model(**encoded_input)
                
                # Process scores
                scores = output[0].cpu().numpy()
                scores = softmax(scores, axis=1)
                
                # Store results
                for score in scores:
                    results.append({
                        'negative': float(score[0]),
                        'neutral': float(score[1]),
                        'positive': float(score[2]),
                        'label': self.config.id2label[score.argmax()],
                        'score': float(score.max()),
                        'raw_outputs': score.tolist()
                    })
            except Exception as e:
                print(f"Error in batch_analyze: {str(e)}")
                results.append({
                    'negative': 0.0,
                    'neutral': 1.0,
                    'positive': 0.0, 
                    'label': 'neutral',
                    'score': 1.0,
                    'raw_outputs': [0.0, 1.0, 0.0]
                })
        
        return results

# Initialize the analyzer as a global instance
_sentiment_analyzer = None

def get_analyzer():
    """Get or create the sentiment analyzer instance"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = NFLSentimentAnalyzer()
    return _sentiment_analyzer

def analyze_sentiment(tweet: str) -> Dict[str, any]:
    """
    Enhanced sentiment analysis using RoBERTa and sports context
    """
    try:
        # Get the analyzer instance
        analyzer = get_analyzer()
        
        # Get RoBERTa sentiment
        roberta_sentiment = analyzer.get_sentiment_scores(tweet)
        
        return roberta_sentiment

    except Exception as e:
        print(f"Error in analyze_sentiment: {str(e)}")
        return {
            'negative': 0.0,
            'neutral': 1.0,
            'positive': 0.0,
            'label': 'neutral', 
            'score': 1.0,
            'raw_outputs': [0.0, 1.0, 0.0]
        }