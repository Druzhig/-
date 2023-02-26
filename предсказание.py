import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model from file
model = tf.keras.models.load_model('model.h5')

# Set the size of the input image
img_width, img_height = 150, 150

# Load the image and resize it to the required size
img = image.load_img(r'C:\жилет\MAN.jpg', target_size=(img_width, img_height))

# Convert the image to a numpy array and normalize the pixel values
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.

# Make a prediction using the loaded model
prediction = model.predict(x)

# Determine the predicted class based on the output of the model
predicted_class = np.argmax(prediction)

# Print the predicted class label
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
