import pandas as pd
import json
from datetime import datetime
from DataCollection import TwitterDataCollector
from DataPreprocessing import TwitterPreprocessor
from SentimentAnalysis import SentimentAnalyzer
from Visualization_and_analysis import SentimentVisualizer 
import os
from dotenv import load_dotenv


import nltk
nltk.download('punkt')




class TwitterSentimentPipeline:
    def __init__(self, twitter_credentials, hf_token=None):
        self.collector = TwitterDataCollector(**twitter_credentials)
        self.preprocessor = TwitterPreprocessor()
        self.analyzer = SentimentAnalyzer(hf_token=hf_token)
        self.results = None
    
    def run_analysis(self, query, max_tweets=1000, save_results=True):
        print(f"Starting sentiment analysis for query: '{query}'")
        
        # Step 1: Collect tweets
        print("1. Collecting tweets...")
        tweets_df = self.collector.collect_tweets(query, max_tweets)
        print(f"   Collected {len(tweets_df)} tweets")
        if tweets_df.empty or 'text' not in tweets_df.columns:
            print("❌ No valid tweets collected or missing 'text' column.")
            return pd.DataFrame()  # or raise Exception("No valid tweets collected.")

        
        # Step 2: Preprocess tweets
        print("2. Preprocessing tweets...")
        processed_df = self.preprocessor.preprocess_dataframe(tweets_df)
        print(f"   {len(processed_df)} tweets after preprocessing")
        
        # Step 3: Analyze sentiment
        print("3. Analyzing sentiment...")
        sentiment_results = self.analyzer.batch_analyze(
            processed_df['cleaned_text'].tolist(), 
            method='textblob'
        )
        
        # Add results to dataframe
        processed_df['sentiment'] = [r['sentiment'] for r in sentiment_results]
        processed_df['confidence'] = [r.get('confidence', 0) for r in sentiment_results]
        
        self.results = processed_df
        
        # Step 4: Generate visualizations
        print("4. Generating visualizations...")
        visualizer = SentimentVisualizer(processed_df)
        visualizer.plot_sentiment_distribution()
        visualizer.generate_insights()
        
        # Step 5: Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.getenv("OUTPUT_DIR", ".")
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"sentiment_analysis_{query.replace(' ', '')}{timestamp}.csv")
            processed_df.to_csv(filename, index=False)
            print(f"5. Results saved to {filename}")
        
        return processed_df
    
    def export_summary(self, filename=None):
        if self.results is None:
            print("No results to export. Run analysis first.")
            return
        
        if filename is None:
            filename = f"sentiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            'total_tweets': len(self.results),
            'sentiment_distribution': self.results['sentiment'].value_counts().to_dict(),
            'average_confidence': float(self.results['confidence'].mean()),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary exported to {filename}")

# Main execution
if __name__ == "__main__":
    load_dotenv()
    # Configure your credentials 
    twitter_credentials = {
        'bearer_token': os.getenv("BEARER_TOKEN"),
        'api_key': os.getenv("API_KEY"),
        'api_secret': os.getenv("API_SECRET"),
        'access_token': os.getenv("ACCESS_TOKEN"),
        'access_token_secret': os.getenv("ACCESS_TOKEN_SECRET")
    }
    hf_token = os.getenv("HF_TOKEN")
    
    # Initialize pipeline
    pipeline = TwitterSentimentPipeline(
        twitter_credentials=twitter_credentials,
        hf_token=""  # Optional
    )
    
    # Run analysis
    results = pipeline.run_analysis(
        query="Python programming",
        max_tweets=500,
        save_results=True
    )
    
    # Export summary
    pipeline.export_summary()
    
    print("Analysis complete!")