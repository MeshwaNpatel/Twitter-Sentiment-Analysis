import tweepy
import pandas as pd
import time

class TwitterDataCollector:
    def __init__(self, bearer_token, api_key, api_secret, access_token, access_token_secret):
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.setup_api()
    
    def setup_api(self):
        # Setup Twitter API v2 client
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            wait_on_rate_limit=True  # Automatically handles basic rate limits
        )
        
        # Setup v1.1 API for optional future use
        auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True)

    def collect_tweets(self, query, max_tweets=10):
        tweets_data = []
        next_token = None
        collected = 0

        print(f"[INFO] Starting tweet collection for query: '{query}' (max {max_tweets} tweets)")

        while collected < max_tweets:
            try:
                print(f"[DEBUG] Collecting batch... Collected so far: {collected}")

                response = self.client.search_recent_tweets(
                    query=query,
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang'],
                    max_results=min(100, max_tweets - collected),
                    next_token=next_token
                )

                if response.data:
                    for tweet in response.data:
                        if tweet.lang == 'en':
                            tweets_data.append({
                                'id': tweet.id,
                                'text': tweet.text,
                                'created_at': tweet.created_at,
                                'author_id': tweet.author_id,
                                'retweet_count': tweet.public_metrics['retweet_count'],
                                'like_count': tweet.public_metrics['like_count'],
                                'reply_count': tweet.public_metrics['reply_count'],
                            })
                            collected += 1
                            if collected >= max_tweets:
                                break

                    next_token = response.meta.get('next_token')
                    if not next_token:
                        print("[INFO] No more tweets available from API.")
                        break

                else:
                    print("[INFO] No tweets returned in response.")
                    break

            except tweepy.TooManyRequests as e:
                # Handle rate limits with sleep
                reset_time = int(e.response.headers.get('x-rate-limit-reset', time.time() + 900))
                sleep_time = max(reset_time - time.time(), 0)
                print(f"[WARNING] Rate limit hit. Sleeping for {int(sleep_time)} seconds...")
                time.sleep(sleep_time + 5)  # Add buffer
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")
                break

        print(f"[INFO] Finished collecting {len(tweets_data)} tweets.")
        return pd.DataFrame(tweets_data)

# Optional test run (for standalone use)
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()

    collector = TwitterDataCollector(
        bearer_token=os.getenv("BEARER_TOKEN"),
        api_key=os.getenv("API_KEY"),
        api_secret=os.getenv("API_SECRET"),
        access_token=os.getenv("ACCESS_TOKEN"),
        access_token_secret=os.getenv("ACCESS_TOKEN_SECRET")
    )
    df = collector.collect_tweets("python", max_tweets=10)
    print(df.head())
