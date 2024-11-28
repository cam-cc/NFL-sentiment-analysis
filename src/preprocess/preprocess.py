from datetime import datetime
import re
from typing import Dict, List
import emoji
from utils import *

def clean_tweet(tweet):
    tweet = tweet.replace(',', ' ')
    tweet = tweet.replace('\n', ' ')
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = ' '.join(tweet.split())
    return tweet

def analyze_game_context(text: str) -> Dict[str, float]:
    """
    Analyzes game-specific context and returns sentiment modifiers
    """
    context_sentiment = 0.0
    modifiers = {}
    
    # Check for game outcomes
    for outcome, data in GAME_OUTCOMES.items():
        if re.search(data['pattern'], text, re.IGNORECASE):
            context_sentiment += data['base_sentiment']
            modifiers['outcome'] = outcome
    
    # Check for performance terms
    for term, score in PERFORMANCE_TERMS['positive'].items():
        if re.search(rf'\b{term}\b', text, re.IGNORECASE):
            context_sentiment += score
            modifiers['performance_positive'] = term
    
    for term, score in PERFORMANCE_TERMS['negative'].items():
        if re.search(rf'\b{term}\b', text, re.IGNORECASE):
            context_sentiment += score
            modifiers['performance_negative'] = term
    
    # Check for streaks
    for streak_type, pattern in STREAK_PATTERNS.items():
        if match := re.search(pattern, text, re.IGNORECASE):
            streak_value = 0.1 if 'winning' in streak_type else -0.1
            if number_match := re.search(r'\d+', match.group()):
                streak_length = int(number_match.group())
                # Increase impact for longer streaks
                streak_value *= min(streak_length, 5)  # Cap at 5 for reasonable bounds
            context_sentiment += streak_value
            modifiers['streak'] = streak_type
    
    # Check for season context
    for term, score in SEASON_CONTEXT['positive'].items():
        if re.search(rf'\b{term}\b', text, re.IGNORECASE):
            context_sentiment += score
            modifiers['season_positive'] = term
            
    for term, score in SEASON_CONTEXT['negative'].items():
        if re.search(rf'\b{term}\b', text, re.IGNORECASE):
            context_sentiment += score
            modifiers['season_negative'] = term
            
    injury_keywords = [
        'injured', 'injury', 'out', 'hurt', 'surgery', 'ir', 
        'injured reserve', 'acl', 'mcl', 'concussion'
    ]
    
    player_keywords = [
        'qb', 'quarterback', 'player', 'starter', 'star', 'rb', 'wr', 'te',
        'running back', 'wide receiver', 'tight end', 'defensive', 'offensive',
        'roster', 'depth chart',
        # Add specific player references
        'dak', 'prescott'  # You might want to load these from a player names file
    ]
            
    text_lower = text.lower()
    has_injury = any(keyword in text_lower for keyword in injury_keywords)
    has_player = any(keyword in text_lower for keyword in player_keywords)
    
    if has_injury and has_player:
        modifiers['injury_context'] = True
        # Enhanced injury severity check
        severity_indicators = {
            'season': -0.9,
            'months': -0.8,
            'weeks': -0.6,
            'out': -0.5
        }
        
        # Find the most severe modifier
        max_severity = -0.5  # default moderate negative
        for indicator, value in severity_indicators.items():
            if indicator in text_lower:
                max_severity = min(max_severity, value)  # Use min since these are negative values
                
        context_sentiment += max_severity
        modifiers['injury_severity'] = abs(max_severity)  # Store the severity level 
    
    # Draft context check (fixed the modifiers reset bug)
    draft_keywords = [
        'draft pick', 'first overall', 'draft position', 'tank', 'tanking',
        'draft order', 'draft spot', 'top pick', 'draft class'
    ]
    
    is_draft_related = any(keyword in text.lower() for keyword in draft_keywords)
    
    if is_draft_related:
        modifiers['draft_context'] = True
        if any(x in text.lower() for x in ['first overall', 'top pick', 'early pick']):
            context_sentiment -= 0.8  # Strong negative modifier
        else:
            context_sentiment -= 0.5  # Moderate negative modifier
            
    return {
        'context_sentiment': context_sentiment,
        'modifiers': modifiers
    }
    
def is_sales_tweet(text: str) -> bool:
    """Check if tweet is promotional/sales related"""
    sales_keywords = {
        'buy', 'sale', 'discount', 'offer', 'shop', 'store', 'price',
        'ticket', 'merch', 'merchandise', 'autograph', '$', 'ebay',
        'purchase', 'order', 'shipping', 'available', '#ad', 'click',
        'link', 'shop', 'store'
    }
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in sales_keywords)


def preprocess_sports_text(text: str) -> Dict[str, any]:
    """
    Enhanced preprocessing for sports-related text.
    Returns dict with processed text and additional features.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Store original emojis before cleaning
    emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
    emoji_sentiment = sum(SPORTS_EMOJI_SENTIMENT.get(e, 0) for e in emoji_list)
    
    # Clean text (enhance existing clean_tweet function)
    text = clean_tweet(text)
    
    # Normalize team names
    for team_variant, full_name in TEAM_NAMES.items():
        text = re.sub(r'\b' + team_variant + r'\b', full_name.lower(), text)
    
    # Handle hashtags - split camel case
    hashtags = re.findall(r'#(\w+)', text)
    for hashtag in hashtags:
        split_hashtag = re.sub(r'([A-Z])', r' \1', hashtag).strip()
        text = text.replace(f'#{hashtag}', split_hashtag.lower())
    
    # Handle score patterns
    score_patterns = re.findall(r'\b(\d+)-(\d+)\b', text)
    has_score = len(score_patterns) > 0
    
    return {
        'processed_text': text,
        'emoji_sentiment': emoji_sentiment,
        'has_score': has_score,
        'emoji_count': len(emoji_list),
        'original_emojis': emoji_list
    }
    