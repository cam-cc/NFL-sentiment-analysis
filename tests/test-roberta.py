import sys
import os
import emoji
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.roberta import analyze_sentiment

def test_roberta_sentiment():
    """Test the RoBERTa-based sentiment analysis using a variety of test cases"""
    test_cases = [
        "Colts with an incredible comeback victory! 🏆 #ForTheShoe",
        "Another disappointing loss. Defense looked terrible today 😭", 
        "Kickoff in 30 minutes! Ready for some football 🏈",
        "3rd straight win! Playoffs here we come! 💪",
        "Tank season in full effect. Time to rebuild 🤦",
        "With the first overall pick in the 2025 NFL Draft the Dallas Cowboys select...",
        "WITH DAK PRESCOTT OUT FOR AT LEAST 4 MONTHS WHERE DO THE DALLAS COWBOYS GO FROM HERE!? WHAT DOES THE FUTURE LOOK LIKE??",
        "Some MAJOR Philadelphia Eagles injury news before they play the Dallas Cowboys tomorrow For the first time since WEEK 1 AJ Brown DeVonta Smith and Dallas Goedert will be on the field at the same time. #Eagles",
        "Third fastest Bronco to reach 5 000 career receiving yards. #ProBowlVote"
    ]
    
    for text in test_cases:
        result = analyze_sentiment(text)
        print(result)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['label']}")
        print(f"Confidence: {result['score']:.3f}")
        print(f"Raw outputs: {result['raw_outputs']}")
        print("-" * 50)

if __name__ == "__main__":
    test_roberta_sentiment()