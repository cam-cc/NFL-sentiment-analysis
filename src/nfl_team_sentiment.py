import twitter_scrape as ts
from utils.utils import *
from preprocess import *
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import os
username = os.getenv('TWITTER_USERNAME')
password = os.getenv('TWITTER_PASSWORD')

if __name__ == "__main__":
    # Get team info
    teams = ['Green Bay Packers']
    start_date = "2024-09-05"
    end_date = "2024-11-10"
    
    for team in teams:
        print(f"Collecting tweets for {team}...")
        try:
            tweets_df = ts.get_unique_tweets(
                username=username,
                password=password,
                team=team,
                start_date=start_date,
                end_date=end_date,
                max_tweets= 1500
        )
            
        except KeyboardInterrupt:
            print("\nCollection interrupted by user. Saving collected tweets...")
            if 'tweets_df' in locals() and tweets_df is not None:
                filename = f"{team.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}_partial.csv"
                tweets_df.to_csv(filename, index=False)
                print(f"Partial data saved to: {filename}")
        
        if tweets_df is not None and len(tweets_df) > 0:
            filename = f"{team.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
            tweets_df.to_csv(filename, index=False)

            print(f"\nTotal tweets collected: {len(tweets_df)}")
            print("\nSentiment Distribution:")
            sentiment_counts = tweets_df['sentiment'].value_counts()
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(tweets_df)) * 100
                print(f"{sentiment}: {percentage:.1f}%")
            
            print(f"\nData saved to: {filename}")
        else:
            print("No tweets were collected. Please check your credentials and try again.")