from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import text2emotion as te

reviews = []
unprocessed_reviews = []
for filename in ["./positive.txt", "./negative.txt"]:
    text_lines = open(filename, "r").readlines()
    text = ""
    for line in text_lines:
        text += (line[:-1] + " ")
    unprocessed_reviews.append(text)

    tokens = word_tokenize(text.lower())

    no_stop_words = []
    for token in tokens:
        if token not in stopwords.words("english"):
            no_stop_words.append(token)

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in no_stop_words]

    processed_text = " ".join(lemmatized_tokens)
    reviews.append(processed_text)

analyzer = SentimentIntensityAnalyzer()
for review in reviews:
    scores = analyzer.polarity_scores(review)
    sentiment = 1 if scores["pos"] > 0 else 0
    print(scores, sentiment)

for review in unprocessed_reviews:
    scores = analyzer.polarity_scores(review)
    sentiment = 1 if scores["pos"] > 0 else 0
    print(scores, sentiment)
# ====================== up: vader, down: t2e
for review in reviews:
    result = te.get_emotion(review)
    print(result)

for review in unprocessed_reviews:
    result = te.get_emotion(review)
    print(result)
