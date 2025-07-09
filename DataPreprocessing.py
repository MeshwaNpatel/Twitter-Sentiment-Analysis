import re
import string
from textblob import TextBlob

class TwitterPreprocessor:
    def __init__(self):
        self.emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       "]+", flags=re.UNICODE)
    
    def clean_tweet(self, tweet):
        # Convert to lowercase
        tweet = tweet.lower()
        
        # Remove URLs
        tweet = re.sub(r'http\S+|www\S+|https\S+', 'http', tweet, flags=re.MULTILINE)
        
        # Remove user mentions and replace with @user
        tweet = re.sub(r'@\w+', '@user', tweet)
        
        # Remove hashtags but keep the text
        tweet = re.sub(r'#(\w+)', r'\1', tweet)
        
        # Remove extra whitespace
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        
        # Remove punctuation except emoticons
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        
        return tweet
    
    def preprocess_dataframe(self, df):
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean tweets
        processed_df['cleaned_text'] = processed_df['text'].apply(self.clean_tweet)
        
        # Remove very short tweets (less than 3 words)
        processed_df = processed_df[processed_df['cleaned_text'].str.split().str.len() >= 3]
        
        # Remove duplicates
        processed_df = processed_df.drop_duplicates(subset=['cleaned_text'])
        
        return processed_df.reset_index(drop=True)

# Usage
# preprocessor = TwitterPreprocessor()
# processed_tweets = preprocessor.preprocess_dataframe(tweets_df)
