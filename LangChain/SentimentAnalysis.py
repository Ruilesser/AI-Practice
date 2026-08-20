from transformers import pipeline

# Load sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    response = sentiment_pipeline(text)
    sentiment = response[0]['label'] # positive or negative
    confidence = response[0]['score']
    return sentiment, confidence

def main():
    print("This is a Sentiment Analysis bot")
    while True:
        question = input("Enter text to analyze ('exit' to quit):")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        sentiment, confidence = analyze_sentiment(question)
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")

main()