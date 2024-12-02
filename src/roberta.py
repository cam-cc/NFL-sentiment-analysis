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
        # Clean and normalize text
        text = clean_tweet(text)
        
        # Handle special tokens
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
            
            temperature = 0.5
            scores = softmax(scores / temperature)
            
            # Get game context modifiers
            context = analyze_game_context(text)
            context_sentiment = context['context_sentiment']
            
            # Apply context bias
            if context_sentiment != 0:
                # Adjust scores based on context
                if context_sentiment > 0:
                    scores[2] *= (1 + context_sentiment)
                    scores[0] *= (1 - context_sentiment)
                else:
                    scores[0] *= (1 - context_sentiment)
                    scores[2] *= (1 + context_sentiment)
                    
                # negative bias for injury context
                if context['modifiers'].get('injury_context'):
                    scores[0] *= 1.5
                    scores[2] *= 0.2
                    scores[1] *= 0.2
                    
                    # negative sentiment for severe injuries
                    if context['modifiers'].get('injury_severity', 0) > 1.0:
                        scores[0] *= 1.5
                        scores[1] *= 0.5
                        scores[2] *= 0.5
                
                # Draft context
                if context['modifiers'].get('draft_context'):
                    if context['modifiers'].get('high_draft_pick'):
                        scores[0] *= 3.0
                        scores[1] *= 0.3
                        scores[2] *= 0.4
                    elif context['modifiers'].get('late_draft_pick'):
                        scores[2] *= 2.0
                
                scores = np.clip(scores / scores.sum(), 0, 1)
            
            return {
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2]),
                'label': self.config.id2label[scores.argmax()],
                'score': float(scores.max()),
                'raw_outputs': scores.tolist(),
                'context_modifiers': context['modifiers']
            }
            
        except Exception as e:
            print(f"Error in get_sentiment_scores: {str(e)}")
            return None
        
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
        analyzer = get_analyzer()
        sentiment_result = analyzer.get_sentiment_scores(tweet)
        return sentiment_result
    except Exception as e:
        print(f"Error in analyze_sentiment: {str(e)}")
        return None