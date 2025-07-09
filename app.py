import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import os
from dotenv import load_dotenv
from main import TwitterSentimentPipeline


# Load environment variables
load_dotenv()

# Load credentials from .env
twitter_credentials = {
    'bearer_token': os.getenv("BEARER_TOKEN"),
    'api_key': os.getenv("API_KEY"),
    'api_secret': os.getenv("API_SECRET"),
    'access_token': os.getenv("ACCESS_TOKEN"),
    'access_token_secret': os.getenv("ACCESS_TOKEN_SECRET")
}
hf_token = os.getenv("HF_TOKEN")

# Import modules

# Streamlit page setup
st.set_page_config(
    page_title="Twitter Sentiment Analysis Tool",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Header
st.markdown('<h1 class="main-header">🐦 Twitter Sentiment Analysis Tool</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Initialize pipeline
if st.sidebar.button("🔧 Initialize Pipeline"):
    try:
        st.session_state.pipeline = TwitterSentimentPipeline(
            twitter_credentials=twitter_credentials,
            hf_token=hf_token
        )
        st.sidebar.success("✅ Pipeline initialized successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error initializing pipeline: {e}")

# Main
if st.session_state.pipeline is None:
    st.info("👆 Please click 'Initialize Pipeline' in the sidebar to get started.")
else:
    st.header("📊 Run Sentiment Analysis")
    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_input("🔍 Enter search query:", placeholder="e.g., Python, Tesla, Bitcoin")

    with col2:
        max_tweets = st.number_input("📈 Max tweets:", min_value=10, max_value=10, value=10, step=10)

    if st.button("🚀 Run Analysis", type="primary"):
        if query:
            with st.spinner("🔄 Analyzing tweets..."):
                try:
                    results = st.session_state.pipeline.run_analysis(
                        query=query,
                        max_tweets=max_tweets,
                        save_results=False
                    )
                    if results.empty or 'text' not in results.columns:
                        st.warning("⚠️ No valid tweets collected. Try a different query.")
                    else:
                        st.session_state.results = results
                        st.session_state.analysis_complete = True
                        st.success(f"✅ Analysis complete! {len(results)} tweets analyzed.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.warning("⚠️ Enter a search query.")

# Results
if st.session_state.analysis_complete and st.session_state.results is not None:
    st.header("📈 Analysis Results")
    results = st.session_state.results

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    sentiment_counts = results['sentiment'].value_counts()
    total_tweets = len(results)
    avg_confidence = results['confidence'].mean()

    with col1:
        st.metric("Total Tweets", total_tweets)
    with col2:
        st.metric("Positive %", f"{(sentiment_counts.get('positive', 0)/total_tweets)*100:.1f}%")
    with col3:
        st.metric("Negative %", f"{(sentiment_counts.get('negative', 0)/total_tweets)*100:.1f}%")
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")

    # Sentiment distribution
    st.subheader("📊 Sentiment Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={
                'positive': '#45b7d1',
                'neutral': '#96ceb4',
                'negative': '#ff6b6b'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title="Sentiment Counts",
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': '#45b7d1',
                'neutral': '#96ceb4',
                'negative': '#ff6b6b'
            }
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Confidence histogram
    st.subheader("🎯 Confidence Distribution")
    fig_conf = px.histogram(
        results,
        x='confidence',
        color='sentiment',
        nbins=20,
        title="Confidence Score Distribution by Sentiment",
        color_discrete_map={
            'positive': '#45b7d1',
            'neutral': '#96ceb4',
            'negative': '#ff6b6b'
        }
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # Engagement metrics
    if 'like_count' in results.columns:
        st.subheader("💬 Engagement by Sentiment")
        col1, col2, col3 = st.columns(3)

        with col1:
            fig_likes = px.box(results, x='sentiment', y='like_count', color='sentiment', title="Likes by Sentiment",
                               color_discrete_map={'positive': '#45b7d1', 'neutral': '#96ceb4', 'negative': '#ff6b6b'})
            fig_likes.update_layout(yaxis_type='log')
            st.plotly_chart(fig_likes, use_container_width=True)

        with col2:
            fig_retweets = px.box(results, x='sentiment', y='retweet_count', color='sentiment',
                                  title="Retweets by Sentiment",
                                  color_discrete_map={'positive': '#45b7d1', 'neutral': '#96ceb4', 'negative': '#ff6b6b'})
            fig_retweets.update_layout(yaxis_type='log')

            st.plotly_chart(fig_retweets, use_container_width=True)

        with col3:
            fig_replies = px.box(results, x='sentiment', y='reply_count', color='sentiment',
                                 title="Replies by Sentiment",
                                 color_discrete_map={'positive': '#45b7d1', 'neutral': '#96ceb4', 'negative': '#ff6b6b'})
            fig_replies.update_layout(yaxis_type='log')
            st.plotly_chart(fig_replies, use_container_width=True)

    # Sample tweets
    st.subheader("📝 Sample Tweets by Sentiment")
    tabs = st.tabs(["😊 Positive", "😐 Neutral", "😞 Negative"])

    for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
        with tabs[i]:
            subset = results[results['sentiment'] == sentiment]
            if not subset.empty:
                top = subset.nlargest(5, 'confidence')
                for _, row in top.iterrows():
                    with st.expander(f"Confidence: {row['confidence']:.3f}"):
                        st.write(row['text'])
                        if 'created_at' in row:
                            st.caption(f"Posted: {row['created_at']}")
            else:
                st.info(f"No {sentiment} tweets found.")

    # Download section
    st.subheader("💾 Download Results")
    col1, col2 = st.columns(2)

    with col1:
        csv = results.to_csv(index=False)
        st.download_button("📄 Download CSV", csv, f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

    with col2:
        summary = {
            'total_tweets': len(results),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'average_confidence': float(avg_confidence),
            'timestamp': datetime.now().isoformat()
        }
        json_str = json.dumps(summary, indent=2)
        st.download_button("📊 Download Summary JSON", json_str, f"sentiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")

# Single text analysis
st.header("🔍 Analyze Single Text")
single_text = st.text_area("Enter text to analyze:")

if st.button("🎯 Analyze Text") and single_text and st.session_state.pipeline:
    try:
        result = st.session_state.pipeline.analyzer.analyze_with_textblob(single_text)
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Sentiment", result['sentiment'].title())
            st.metric("Confidence", f"{result['confidence']:.3f}")

        with col2:
            df = pd.DataFrame([{'Sentiment': k.title(), 'Score': v} for k, v in result['scores'].items()])
            fig = px.bar(df, x='Sentiment', y='Score', color='Sentiment', title="Sentiment Scores",
                         color_discrete_map={'Positive': '#45b7d1', 'Neutral': '#96ceb4', 'Negative': '#ff6b6b'})
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error analyzing text: {e}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Twitter Sentiment Analysis Tool")
