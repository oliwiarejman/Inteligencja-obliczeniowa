# Load LSTM network and generate text
import sys
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from keras.src.models import Sequential
from keras.src.saving import load_model
from keras.src.layers import Dense
from keras.src.layers import Dropout
from keras.src.layers import LSTM
from keras.src.callbacks import ModelCheckpoint
from keras.src.utils import to_categorical
# load ascii text and covert to lowercase
filename = "text.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
tokenized_text = wordpunct_tokenize(raw_text)
tokens = sorted(list(dict.fromkeys(tokenized_text)))

#print("Tokens: ")
#print(tokens)
tok_to_int = dict((c, i) for i, c in enumerate(tokens))
int_to_tok = dict((i, c) for i, c in enumerate(tokens))
#print("TokensToNumbers: ")
#print(tok_to_int)

# summarize the loaded data
n_tokens = len(tokenized_text)
n_token_vocab = len(tokens)
print("Total Tokens: ", n_tokens)
print("Unique Tokens (Token Vocab): ", n_token_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_tokens - seq_length, 1):
	seq_in = tokenized_text[i:i + seq_length]
	seq_out = tokenized_text[i + seq_length]
	dataX.append([tok_to_int[tok] for tok in seq_in])
	dataY.append(tok_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_token_vocab)
# one hot encode the output variable
y = to_categorical(dataY)
# define the LSTM model
# model = load_model("./big-token-model-40-3.6852.keras")
model = load_model("./big-token-model-40-5.3865.keras")
# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ' '.join([int_to_tok[value] for value in pattern]), "\"")
# generate tokens
print("Generated text:")
for i in range(100):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_token_vocab)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = int_to_tok[index]
	seq_in = [int_to_tok[value] for value in pattern]
	sys.stdout.write(result+" ")
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")
