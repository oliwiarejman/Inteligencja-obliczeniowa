import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string

# Pobranie niezbędnych zasobów NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Wczytanie artykułu z pliku
with open('article.txt', 'r') as file:
    article = file.read()

# Usunięcie znaków interpunkcyjnych i cyfr
article = ''.join([char for char in article if char not in string.punctuation and not char.isdigit()])

# Tokenizacja artykułu
tokens = word_tokenize(article)
num_words_tokenized = len(tokens)
print(f'Liczba słów po tokenizacji: {num_words_tokenized}')

# Pobranie listy stop-words
stop_words = set(stopwords.words('english'))

# Usuwanie stop-words
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
num_words_filtered = len(filtered_tokens)
print(f'Liczba słów po usunięciu stop-words: {num_words_filtered}')

# Dodanie dodatkowych stop-words
additional_stopwords = ['The', 'Met', 'Office', 'NASA', 'NOAA', 'WMO', 'US', 'World', 'says', 'saying', 'year', 'years']
stop_words.update(additional_stopwords)

# Ponowne usuwanie stop-words
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
num_words_filtered_again = len(filtered_tokens)
print(f'Liczba słów po dodatkowym czyszczeniu: {num_words_filtered_again}')

# Inicjalizacja lematyzera
lemmatizer = WordNetLemmatizer()

# Lematyzacja
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
num_words_lemmatized = len(lemmatized_tokens)
print(f'Liczba słów po lematyzacji: {num_words_lemmatized}')

# Zliczanie słów
word_counts = Counter(lemmatized_tokens)

# Wybór 10 najczęstszych słów
most_common_words = word_counts.most_common(10)
words, counts = zip(*most_common_words)

# Wykres słupkowy
plt.figure(figsize=(10, 5))
plt.bar(words, counts)
plt.xlabel('Słowa')
plt.ylabel('Liczba wystąpień')
plt.title('10 najczęściej występujących słów')
plt.show()

# Stworzenie chmury tagów
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

# Wyświetlenie chmury tagów
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
