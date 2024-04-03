import numpy as np
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# standard scaler zaweza zakres danych do wartosci z przedzialu np. -1 ; 1
# niektore dane sa z zakresu 10 do 50, a inne -30 do -20, co moze byc w nieprawidlowy sposob zinterpretowane
# standard scaler ustawia pewna skale pomiedzy tymi danymi by wszystkie byly w miare jednolite i latwiejsze do interpretacji

# Encode the labels
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))
# onehot encoding pozwala na zdefiniowanie danych kategorycznych jako zestaw kolumn z danymi numerycznymi,
# prowadzi to do zmniejszonego biasu modelu np. dane z kolumny Sex: Male/Female zostana rozbite na Male | Female
#                                                                                                   0/1     0/1        
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='tanh', input_shape=(X_train.shape[1],)),
    Dense(64, activation='tanh'),
    Dense(y_encoded.shape[1], activation='softmax')
])
# warstwa wejsciowa ma 4 neurony, shape nadaje ksztalt danym, x_train.shape ma 4 elementy, 
# a y_encoded ma 3 (kolumny setosa virginica i versicolor) z wartosciami 1 lub 0
# Compile the model
# adamw pogarsza, adamax sobie radzi, sgd slabo, nadam mega dobrze 
# mean squared error mega fajnie
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# dla wiekszego batch size siec lepiej sie uczyla
# wiecej epok niz 100 nie ma sensu bo siec sie nie uczy dalej
# shuffle chyba nieco poprawia nauke sieci
history = model.fit(X_train, y_train, epochs=100, shuffle=True, validation_split=0.2, batch_size=16)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
# wydaje mi sie ze siec sie dobrze nauczyla, najwieksza wydajnosc jest w okolicach 95-100 epoki
# moze byc troszke niedouczony? 
# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model

# Plot and save the model architecture

