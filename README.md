# Twitter-Sentiment-Analysis
A full-stack sentiment analysis project that fetches real-time tweets using the Twitter API, classifies them using NLP models, and displays the results in an interactive dashboard powered by Streamlit.


## 🚀 Features

- 🔍 Collect real-time tweets via Twitter API
- 🧹 Clean and preprocess tweets using NLP techniques
- 🤖 Analyze sentiments using TextBlob and Hugging Face RoBERTa
- 📊 Interactive charts and visualizations (Pie, Bar, Histogram)
- 📝 Sample tweet viewer by sentiment category
- 💾 Export data as CSV and summary as JSON
- 🐳 Dockerized for easy deployment

---

## 🛠️ Technologies Used

| Technology     | Purpose                                 |
|----------------|-----------------------------------------|
| Python         | Programming language                    |
| Tweepy         | Twitter API integration                 |
| TextBlob       | Basic sentiment analysis                |
| Hugging Face   | Transformer-based sentiment analysis    |
| Pandas         | Data handling and preprocessing         |
| Streamlit      | Web app interface                       |
| Plotly         | Visualizations                          |
| Matplotlib     | Additional visual insights              |
| Docker         | Containerized deployment                |

---

## ⚙️ Setup Instructions

### 1. Add your Twitter and Hugging Face credentials

Create a .env file with the following structure:

.env:

BEARER_TOKEN=your_bearer_token
API_KEY=your_api_key
API_SECRET=your_api_secret
ACCESS_TOKEN=your_access_token
ACCESS_TOKEN_SECRET=your_access_token_secret
HF_TOKEN=your_huggingface_token


### 2. Install dependencies (if running locally):

pip install -r requirements.txt

### 3. Run the Streamlit app:

streamlit run streamlit.py


## 🐳 Running with Docker

To run the app inside a Docker container:

### 1. Build the Docker image

 use Docker Compose:

docker-compose up --build

## 🧠 Future Enhancements

- Real-time tweet streaming
- Support for multiple languages
- Fine-tuned custom models
- Save outputs to a database


## 🙌 Acknowledgments

- Hugging Face Transformers
- Streamlit Community
- Twitter Developer Portal
