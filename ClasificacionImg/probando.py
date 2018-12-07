import numpy as np
#import cnn
from keras.models import load_model
from keras.preprocessing import image

classifier = load_model("animalsPesos.h5")
print("Cargado modelo desde disco.")
# Compilar modelo cargado y listo para usar.
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
  
test_image = image.load_img('ImgPrueba/perro2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

print(result)
print(result[0])
print(result[0][0])
if result[0][0] == 0:
    prediction = 'gato'
if result[0][0] == 1:
    prediction = 'perro'

print('Prediccion: ', prediction)