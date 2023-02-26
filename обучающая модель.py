import tensorflow as tf
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# задаем параметры модели
input_shape = (150, 150, 3)
num_classes = 2

# создаем модель
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# компилируем модель
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# загружаем данные для обучения
train_datagen = ImageDataGenerator(rescale=1./255)



train_df = pd.DataFrame({
        'filename': ['MAN.jpg'], # имя файла
        'class': ['class_name'] # имя класса
})

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory='C:/жилет', # директория, в которой находится файл
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)


# обучаем модель
model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30
)

# сохраняем модель в файл
model.save('model.h5')
