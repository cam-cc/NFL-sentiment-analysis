import sys
import os
import emoji
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.roberta import analyze_sentiment

def test_roberta_sentiment():
    """Test the enhanced RoBERTa-based sentiment analysis"""
    test_cases = [
        "Colts with an incredible comeback victory! 🏆 #ForTheShoe",
        "Another disappointing loss. Defense looked terrible today 😭",
        "Kickoff in 30 minutes! Ready for some football 🏈",
        "3rd straight win! Playoffs here we come! 💪",
        "Tank season in full effect. Time to rebuild 🤦",
        "With the first overall pick in the 2025 NFL Draft the Dallas Cowboys select...",
        "WITH DAK PRESCOTT OUT FOR AT LEAST 4 MONTHS WHERE DO THE DALLAS COWBOYS GO FROM HERE!? WHAT DOES THE FUTURE LOOK LIKE??",
        "Some MAJOR Philadelphia Eagles injury news before they play the Dallas Cowboys tomorrow For the first time since WEEK 1 AJ Brown DeVonta Smith and Dallas Goedert will be on the field at the same time. #Eagles"
    ]
    
    for text in test_cases:
        result = analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment_label']}")
        print(f"Compound Score: {result['compound']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"RoBERTa Scores:", result['sentiment_details']['roberta_scores'])
        print(f"Context:", result['context'])
        print("-" * 50)

if __name__ == "__main__":
    test_roberta_sentiment()