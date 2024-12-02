from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
from datetime import datetime
import re
from preprocess import *
from roberta import *

def twitter_login(driver, username, password):
    driver.get("https://twitter.com/login")
    time.sleep(3)
    
    try:
        # Enter username
        username_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@autocomplete='username']"))
        )
        username_field.send_keys(username)
        username_field.send_keys(Keys.RETURN)
        time.sleep(2)
        
        # Enter password
        password_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@name='password']"))
        )
        password_field.send_keys(password)
        password_field.send_keys(Keys.RETURN)
        time.sleep(3)
        
        return True
        
    except Exception as e:
        print(f"Login error: {e}")
        return False
    
def scroll_with_delay(driver, pause_time=2.0):
    """Controlled scrolling with dynamic delay"""
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.5);")
        time.sleep(pause_time/2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        pause_time = min(pause_time * 1.2, 5.0) 

def get_unique_tweets(username, password, team, start_date, end_date, max_tweets=100):
    
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    driver = webdriver.Chrome(options=options)
   
    # login 
    if not twitter_login(driver, username, password):
        driver.quit()
        return None
    
    # Search Query 
    search_url = f"https://twitter.com/search?q={team}%20lang%3Aen%20-filter:videos%20-filter:media%20-filter:replies%20-filter:retweets%20since%3A{start_date}%20until%3A{end_date}&src=typed_query&f=live"
    driver.get(search_url)
    time.sleep(3)
    
    seen_tweets = set()
    tweets = []
    print(f"\nStarting collection for {team}. Target: {max_tweets} tweets")
    
    while len(tweets) < max_tweets:
        try:
            tweet_elements = WebDriverWait(driver, 20).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="tweet"]'))
            )
            for tweet in tweet_elements:
                if len(tweets) >= max_tweets:
                    break
                
                try:
                    text = tweet.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]').text
                    
                    # avoid replies and retweets (duplicates)
                    if (text.startswith('@') or 
                        text.startswith('RT @') or 
                        'Replying to @' in text):
                        continue
                    
                    cleaned_text = clean_tweet(text)
                    
                    if is_sales_tweet(cleaned_text):
                        print("Skipping promotional/sales tweet...")
                        continue                        
                    
                    if cleaned_text not in seen_tweets:
                        user_info = tweet.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
                        username = user_info.text.split('\n')[0]
                        try:
                            time_element = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.TAG_NAME, "time"))
                            )
                            timestamp = time_element.get_attribute("datetime")
                        except Exception as e:
                            print("Time element not found, skipping timestamp extraction")
                            timestamp = None
                            
                        # sentiment analysis with RoBERTa
                        sentiment_scores = analyze_sentiment(cleaned_text)
                        
                        tweet_data = {
                            'username': username,
                            'timestamp': timestamp,
                            'text': cleaned_text,
                            'team': team,
                            'sentiment': sentiment_scores['label'],
                            'confidence': sentiment_scores['score'],
                            'roberta_raw_outputs': sentiment_scores['raw_outputs']
                        } 
                        
                        BATCH_SIZE = 20 
                        BATCH_DELAY = 3 

                        if len(tweets) % BATCH_SIZE == 0 and len(tweets) > 0:
                            print(f"Batch complete. Collected {len(tweets)} tweets. Pausing...")
                            time.sleep(BATCH_DELAY)
                        
                        tweets.append(tweet_data)
                        seen_tweets.add(cleaned_text)
                        print(f"Successfully collected tweet from {username}. Total: {len(tweets)}/{max_tweets}")
                        
                        # Save progress periodically
                        if len(tweets) % 100 == 0:
                            temp_df = pd.DataFrame(tweets)
                            temp_df.to_csv(f"{team.replace(' ', '_')}_temp.csv", index=False)
                            print(f"Saved {len(tweets)} tweets to temporary file")
                        
                except Exception as e:
                    print(f"Error processing tweet: {str(e)}")
                    continue
            
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            break
    
    print(f"\nCollection complete. Total tweets collected: {len(tweets)}")
    
    driver.quit()
    return pd.DataFrame(tweets) 