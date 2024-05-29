import cv2
import os
import numpy as np

from keras.src.applications import VGG16
from keras.src.models import Sequential
from keras.src.layers import Dense, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator

szerokosc, wysokosc = 224, 224

sciezka_folderu = 'dogs-cat-mini'

generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

batch_size = 32

generator_treningowy_flow = generator.flow_from_directory(
        sciezka_folderu,
        target_size=(szerokosc, wysokosc),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

generator_walidacyjny_flow = generator.flow_from_directory(
        sciezka_folderu,
        target_size=(szerokosc, wysokosc),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation')

vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(szerokosc, wysokosc, 3))

model = Sequential()
model.add(vgg16_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

vgg16_base.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

liczba_epok = 10
historia = model.fit(generator_treningowy_flow, 
                     steps_per_epoch=len(generator_treningowy_flow),
                     epochs=liczba_epok,
                     validation_data=generator_walidacyjny_flow,
                     validation_steps=len(generator_walidacyjny_flow))


model.save('model_psow_i_kotow.h5')


wyniki = model.evaluate(generator_walidacyjny_flow)
print("Dokładność modelu na danych walidacyjnych: %.2f%%" % (wyniki[1] * 100))
