import numpy as np
from tensorflow.keras.callbacks import History
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import seed, random
from os import listdir, makedirs
from shutil import copyfile
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
import seaborn as sns

FAST_RUN = False
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
epochs = 100

folder = 'train/'

subdirs = ['train/', 'test/']
for subdir in subdirs:
    labdirs = ['cats/', 'dogs/']
    for labdir in labdirs:
        newdir = 'data/' + subdir + labdir
        makedirs(newdir, exist_ok=True)

seed(1)
val_ratio = 0.3

for file in listdir(folder):
    src = folder + file
    dest_dir = "data/train/"
    if random() < val_ratio:
        dest_dir = "data/test/"
    if file.startswith("cat"):
        dest = dest_dir + "cats/" + file
    elif file.startswith("dog"):
        dest = dest_dir + "dogs/" + file
    copyfile(src, dest)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(
        IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation="softmax")
])
# opt = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-07)
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

datagen = ImageDataGenerator(rescale=1.0/255.0,
                             rotation_range=20,
                             shear_range=0.11,
                             zoom_range=0.22,
                             horizontal_flip=True,
                             width_shift_range=0.12,
                             height_shift_range=0.12)
batch_size = 32
train_it = datagen.flow_from_directory(
    "./data/train/", class_mode="categorical", shuffle=True, batch_size=batch_size, target_size=(200, 200))
validation_it = datagen.flow_from_directory(
    "./data/test/", class_mode="categorical", shuffle=True, batch_size=batch_size, target_size=(200, 200))
test_it = datagen.flow_from_directory(
    "./data/test/", class_mode="categorical", shuffle=True, batch_size=1, target_size=(200, 200))

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy",
    patience=2,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)
callbacks = [learning_rate_reduction]
history = model.fit(train_it,
                    steps_per_epoch=len(train_it)//batch_size,
                    validation_data=validation_it,
                    validation_steps=len(validation_it)//batch_size,
                    epochs=epochs,
                    callbacks=callbacks)

predictions = model.predict(test_it, steps=len(test_it))
predict_labels = np.argmax(predictions, axis=1)
labels = test_it.labels
model.save("cat_vs_dog_model.keras")
# Confusion matrix
cm = confusion_matrix(labels, predict_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()
