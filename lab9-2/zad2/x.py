import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


positive_review = ("I had an amazing stay at this hotel. The staff was incredibly friendly and went out of their way "
                   "to make our stay comfortable. The room was clean and spacious, and the view from the balcony was breathtaking. "
                   "The food at the hotel restaurant was delicious. I would definitely recommend this hotel to anyone visiting the area.")


negative_review = ("My stay at this hotel was terrible. The room was dirty and smelled bad. The staff was rude and unhelpful. "
                   "The Wi-Fi was non-existent and the food was awful. I would never stay at this hotel again and advise others to avoid it.")

positive_scores = sia.polarity_scores(positive_review)
negative_scores = sia.polarity_scores(negative_review)

positive_blob = TextBlob(positive_review)
negative_blob = TextBlob(negative_review)

print("Vader Sentiment Analysis:")
print("Positive Review Scores:", positive_scores)
print("Negative Review Scores:", negative_scores)

print("\nTextBlob Emotion Analysis:")
print("Positive Review Emotions:", positive_blob.sentiment)
print("Negative Review Emotions:", negative_blob.sentiment)
