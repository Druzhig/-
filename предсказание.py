import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Загружаем модель
model = tf.keras.models.load_model('model.h5')

# Задаем размер
img_width, img_height = 150, 150

# Загружаем изображение и применяем размер
img = image.load_img(r'C:\жилет\MAN.jpg', target_size=(img_width, img_height))

# Конвертируем в Numpy массив и нормализация значение пикселей
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.

# Сделать прогноз, использую загруженную модель
prediction = model.predict(x)

# Определяем прогнозируемый класс на основе выходных данных моделиl
predicted_class = np.argmax(prediction)

# Выводим результат
if predicted_class == 0:
    print('Человек в светоотражающем жилете')
else:
    print('Человек в строительной каске')

# выводим результат
if prediction[0][0] > prediction[0][1]:
    print('Человек в светоотражающем жилете')
else:
    print('Человек в строительной каске')
#Этот код предсказывает
