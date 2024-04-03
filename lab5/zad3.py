import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

data_dir = 'dogs-cats-mini'

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.startswith("cat"):
            label = 0
        else:
            label = 1
        img = imread(os.path.join(folder, filename))
        img = resize(img, (150, 150)) 
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

images, labels = load_images(data_dir)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

y_pred = model.predict_classes(X_test)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))

target_names = ['Cat', 'Dog']
print('Classification Report')
print(classification_report(y_test, y_pred, target_names=target_names))
