SPORTS_STOPWORDS = {
    'yard', 'quarter', 'touchdown', 'field', 'goal', 'pass', 'run',
    'game', 'score', 'playoff', 'season', 'snap', 'drive', 'ball', 'tackle',
    'sack', 'punt', 'kick', 'fumble', 'interception', 'offense', 'defense',
    'receiver', 'quarterback', 'running back', 'linebacker', 'safety', 'cornerback',
    'kicker', 'punter', 'rushing', 'passing', 'coverage', 'blocking', 'penalty',
    'first down', 'turnover', 'huddle', 'scrimmage', 'endzone', 'sideline',
    'playbook', 'roster', 'depth chart', 'formation', 'blitz', 'screen'
}

# Team name variations
TEAM_NAMES = {
    'colts': 'Indianapolis Colts',
    'indy': 'Indianapolis Colts',
    'packers': 'Green Bay Packers',
    'pack': 'Green Bay Packers',
    'broncos': 'Denver Broncos',
    'denver': 'Denver Broncos',
    'bills': 'Buffalo Bills',
    'buffalo': 'Buffalo Bills',
    'dolphins': 'Miami Dolphins',
    'miami': 'Miami Dolphins',
    'patriots': 'New England Patriots',
    'ne': 'New England Patriots',
    'new england': 'New England Patriots',
    'ravens': 'Baltimore Ravens',
    'baltimore': 'Baltimore Ravens',
    'bengals': 'Cincinnati Bengals',
    'cincinnati': 'Cincinnati Bengals',
    'steelers': 'Pittsburgh Steelers',
    'pittsburgh': 'Pittsburgh Steelers',
    'texans': 'Houston Texans',
    'houston': 'Houston Texans',
    'titans': 'Tennessee Titans',
    'tennessee': 'Tennessee Titans',
    'jaguars': 'Jacksonville Jaguars',
    'jacksonville': 'Jacksonville Jaguars',
    'chiefs': 'Kansas City Chiefs',
    'kc': 'Kansas City Chiefs',
    'kansas city': 'Kansas City Chiefs',
    'raiders': 'Las Vegas Raiders',
    'las vegas': 'Las Vegas Raiders',
    'lv': 'Las Vegas Raiders',
    'chargers': 'Los Angeles Chargers',
    'lac': 'Los Angeles Chargers',
    'la chargers': 'Los Angeles Chargers',
    'rams': 'Los Angeles Rams',
    'lar': 'Los Angeles Rams',
    'la rams': 'Los Angeles Rams',
    'cardinals': 'Arizona Cardinals',
    'arizona': 'Arizona Cardinals',
    'falcons': 'Atlanta Falcons',
    'atlanta': 'Atlanta Falcons',
    'cowboys': 'Dallas Cowboys',
    'dallas': 'Dallas Cowboys',
    'giants': 'New York Giants',
    'nyg': 'New York Giants',
    'new york giants': 'New York Giants',
    'eagles': 'Philadelphia Eagles',
    'philadelphia': 'Philadelphia Eagles',
    'redskins': 'Washington Commanders',
    'washington': 'Washington Commanders',
    'lions': 'Detroit Lions',
    'detroit': 'Detroit Lions',
    'packers': 'Green Bay Packers',
    'vikings': 'Minnesota Vikings',
}

# Sports-specific emoji sentiment
SPORTS_EMOJI_SENTIMENT = {
    '🏈': 0.0,    # neutral - football
    '🏃': 0.0,    # neutral - running
    '💪': 0.4,    # positive - strength
    '🔥': 0.4,    # positive - fire/hot
    '😤': 0.3,    # positive in sports context
    '🏆': 0.5,    # very positive
    '😭': -0.4,   # negative
    '💔': -0.4,   # negative
    '🤦': -0.3,   # negative
    '👎': -0.4    # negative
}
# Enhanced sports context dictionaries
GAME_OUTCOMES = {
    'win': {'pattern': r'\b(W|win|won|victory|dub)\b', 'base_sentiment': 0.3},
    'loss': {'pattern': r'\b(L|loss|lost|fell|dropped)\b', 'base_sentiment': -0.3},
}

PERFORMANCE_TERMS = {
    'positive': {
        'clutch': 0.4,
        'mvp': 0.5,
        'elite': 0.4,
        'dominant': 0.4,
        'unstoppable': 0.5,
        'perfect': 0.5,
        'OPOY': 0.5,
        'playoff bound': 0.4,
        'super bowl bound': 0.4,
        'super bowl contender': 0.4,
        'undefeated': 0.5,
        'ball out': 0.4,
    },
    'negative': {
        'choke': -0.4,
        'bust': -0.4,
        'trash': -0.5,
        'terrible': -0.4,
        'awful': -0.4,
        'pathetic': -0.5,
        'concussion': -0.5,
        'injury': -0.5,
        'disappointment': -0.4,
        'fail': -0.4,
        'rebuild': -0.3,
        'losing streak': -0.3,
        'interception': -0.4,
        'turnover': -0.4,
        'IR': -0.4,
        'PUP': -0.4,
        'suspended': -0.4,
        'benched': -0.4,
        'missed': -0.4,
        'out': -0.4,
        'rest': -0.4,
        'cut': -0.4,
        'season ending': -0.4,
        'injury': -0.4,
        'surgery': -0.4,
        'fix': -0.4,
    }
}

STREAK_PATTERNS = {
    'winning_streak': r'\b(\d+)(?:th|rd|nd|st)? straight (?:win|victory)',
    'losing_streak': r'\b(\d+)(?:th|rd|nd|st)? straight (?:loss|L)',
    'undefeated': r'\bundefeated\b',
    'winless': r'\bwinless\b'
}

SEASON_CONTEXT = {
    'positive': {
        'playoff': 0.3,
        'superbowl': 0.4,
        'championship': 0.3,
        'contender': 0.3
    },
    'negative': {
        'eliminated': -0.3,
        'lottery': -0.2,
        'tank': -0.3,
        'rebuild': -0.1
    }
}