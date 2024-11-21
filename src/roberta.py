# Add to utils.py
from .preprocess import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
from scipy.special import softmax
import numpy as np
from functools import lru_cache

class NFLSentimentAnalyzer:
    def __init__(self):
        self.MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)
        self.config = AutoConfig.from_pretrained(self.MODEL)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def preprocess(self, text: str) -> str:
        """Preprocess text for the model"""
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    @lru_cache(maxsize=1024)
    def get_sentiment_scores(self, text: str) -> dict:
        """Get sentiment scores for a single text"""
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
        
        return {
            'negative': float(scores[0]),
            'neutral': float(scores[1]),
            'positive': float(scores[2]),
            'sentiment_label': self.config.id2label[scores.argmax()],
            'confidence': float(scores.max())
        }
    
    def batch_analyze(self, texts: list, batch_size: int = 16) -> list:
        """Analyze sentiment for a batch of texts"""
        results = []
        for i in range(0, len(texts), batch_size):
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
                    'sentiment_label': self.config.id2label[score.argmax()],
                    'confidence': float(score.max())
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
    # Get the analyzer instance
    analyzer = get_analyzer()
    
    # Get sports context features
    context_analysis = analyze_game_context(tweet)
    sports_features = preprocess_sports_text(tweet)
    
    # Get RoBERTa sentiment
    roberta_sentiment = analyzer.get_sentiment_scores(tweet)
    
    # Scale from -1 to 1 for compatibility with previous system
    compound = (roberta_sentiment['positive'] - roberta_sentiment['negative'])
    
    # Increase negative weighting for draft context
    if 'draft_context' in context_analysis['modifiers']:
        # Force negative sentiment for draft-related content
        roberta_sentiment['sentiment_label'] = 'negative'
        compound = min(compound - 0.5, -0.3)  # Ensure it stays negative
        
        # Adjust the scores to reflect negative sentiment
        roberta_sentiment['negative'] = max(roberta_sentiment['negative'], 0.6)
        roberta_sentiment['neutral'] = min(roberta_sentiment['neutral'], 0.3)
        roberta_sentiment['positive'] = min(roberta_sentiment['positive'], 0.1)
        
    
     # Adjust sentiment based on context
    if 'injury_context' in context_analysis['modifiers']:
        severity = context_analysis['modifiers']['injury_severity']
        if severity >= 0.7:  # For severe injuries (long-term, months, season)
            # Force negative sentiment
            roberta_sentiment['sentiment_label'] = 'negative'
            # Dramatically adjust the scores
            roberta_sentiment['negative'] = max(roberta_sentiment['negative'], 0.7)
            roberta_sentiment['neutral'] = min(roberta_sentiment['neutral'] * 0.4, 0.2)
            roberta_sentiment['positive'] = 0.0
            
            # Ensure compound score is significantly negative
            compound = -0.6 - (severity * 0.3)  # Will result in -0.6 to -0.9
    
    # Combine with context
    final_compound = (
        0.7 * compound +  # RoBERTa sentiment
        0.2 * context_analysis['context_sentiment'] +  # Sports context
        0.1 * sports_features['emoji_sentiment']  # Emoji sentiment
    ) 
    
    return {
        # Legacy compatibility
        'compound': final_compound,
        'sentiment_label': roberta_sentiment['sentiment_label'],
        
        # Enhanced features
        'sentiment_details': {
            'roberta_scores': {
                'negative': roberta_sentiment['negative'],
                'neutral': roberta_sentiment['neutral'],
                'positive': roberta_sentiment['positive']
            },
            'context_sentiment': context_analysis['context_sentiment'],
            'emoji_sentiment': sports_features['emoji_sentiment']
        },
        'context': context_analysis['modifiers'],
        'confidence': roberta_sentiment['confidence'],
        'processed_text': sports_features['processed_text']
    }