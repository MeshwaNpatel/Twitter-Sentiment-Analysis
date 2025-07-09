import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

class SentimentVisualizer:
    def __init__(self, df):
        self.df = df
        plt.style.use('seaborn-v0_8')
    
    def plot_sentiment_distribution(self):
        plt.figure(figsize=(10, 6))
        
        # Count plot
        plt.subplot(1, 2, 1)
        sentiment_counts = self.df['sentiment'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Sentiment Distribution')
        
        # Bar plot
        plt.subplot(1, 2, 2)
        sns.countplot(data=self.df, x='sentiment', palette=colors)
        plt.title('Sentiment Counts')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confidence_distribution(self):
        plt.figure(figsize=(12, 4))
        
        for i, sentiment in enumerate(['negative', 'neutral', 'positive']):
            plt.subplot(1, 3, i+1)
            sentiment_data = self.df[self.df['sentiment'] == sentiment]['confidence']
            plt.hist(sentiment_data, bins=20, alpha=0.7, color=['#ff6b6b', '#4ecdc4', '#45b7d1'][i])
            plt.title(f'{sentiment.capitalize()} Confidence')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def create_wordclouds(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sentiments = ['negative', 'neutral', 'positive']
        colors = ['Reds', 'Greys', 'Blues']
        
        for i, sentiment in enumerate(sentiments):
            sentiment_text = ' '.join(self.df[self.df['sentiment'] == sentiment]['cleaned_text'])
            
            if sentiment_text:
                wordcloud = WordCloud(width=400, height=300, 
                                    background_color='white',
                                    colormap=colors[i]).generate(sentiment_text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{sentiment.capitalize()} Words')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_engagement_by_sentiment(self):
        if 'like_count' in self.df.columns:
            plt.figure(figsize=(12, 4))
            
            # Likes by sentiment
            plt.subplot(1, 3, 1)
            sns.boxplot(data=self.df, x='sentiment', y='like_count')
            plt.title('Likes by Sentiment')
            plt.yscale('log')
            
            # Retweets by sentiment
            plt.subplot(1, 3, 2)
            sns.boxplot(data=self.df, x='sentiment', y='retweet_count')
            plt.title('Retweets by Sentiment')
            plt.yscale('log')
            
            # Replies by sentiment
            plt.subplot(1, 3, 3)
            sns.boxplot(data=self.df, x='sentiment', y='reply_count')
            plt.title('Replies by Sentiment')
            plt.yscale('log')
            
            plt.tight_layout()
            plt.show()
    
    def generate_insights(self):
        total_tweets = len(self.df)
        sentiment_counts = self.df['sentiment'].value_counts()
        
        print("=== SENTIMENT ANALYSIS INSIGHTS ===")
        print(f"Total tweets analyzed: {total_tweets}")
        print(f"Positive tweets: {sentiment_counts.get('positive', 0)} ({sentiment_counts.get('positive', 0)/total_tweets*100:.1f}%)")
        print(f"Neutral tweets: {sentiment_counts.get('neutral', 0)} ({sentiment_counts.get('neutral', 0)/total_tweets*100:.1f}%)")
        print(f"Negative tweets: {sentiment_counts.get('negative', 0)} ({sentiment_counts.get('negative', 0)/total_tweets*100:.1f}%)")
        
        if 'confidence' in self.df.columns:
            avg_confidence = self.df['confidence'].mean()
            print(f"Average confidence: {avg_confidence:.3f}")
        
        # Most common words by sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_tweets = self.df[self.df['sentiment'] == sentiment]['cleaned_text']
            if not sentiment_tweets.empty:
                all_words = ' '.join(sentiment_tweets).split()
                common_words = Counter(all_words).most_common(5)
                print(f"\nTop words in {sentiment} tweets:")
                for word, count in common_words:
                    print(f"  {word}: {count}")

# Usage
# visualizer = SentimentVisualizer(processed_tweets)
# visualizer.plot_sentiment_distribution()
# visualizer.plot_confidence_distribution()
# visualizer.create_wordclouds()
# visualizer.plot_engagement_by_sentiment()
# visualizer.generate_insights()
