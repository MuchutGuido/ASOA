# Importando las librerias y paquetes de Keras

# Para hacer un modelo de red neuronal como una red secuencial (secuencia de capas) 
from keras.models import Sequential, model_from_json
# Para realizar operacion de Convolusion
from keras.layers import Conv2D
# Para operacion de agrupacion (agr min, agr media, agr max) MaxPoooling (pixel de valor max de la region)
from keras.layers import MaxPooling2D
# Para convertir matrices bidimensionales en un solo vector lineal
from keras.layers import Flatten
# Para realizar la conexion completa de la red neuronal
from keras.layers import Dense

# Se crea un objeto de la clase Sequential
clasificador = Sequential()

# Se agrega al clasificador la operacion de convolucion
# parametros de Conv2dD(num de filtros, forma de cada filtro, forma de entrada y tipo de imagen[64x64 dimension y 3 RGB], funcion de activacion)
clasificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation='relu')) #capa de entrada

# Para reducir las img lo que mas se pueda 
clasificador.add(MaxPooling2D(pool_size=(2,2)))

# Se convierte a un vector unico todas las img agrupadas
clasificador.add(Flatten())

# Para agregar una capa completamente conectada (units = cantidad de nodos) funcion de activacion
clasificador.add(Dense(units=128, activation='relu')) #Capa oculta

# Nos dara una unica salida
clasificador.add(Dense(units=1, activation='sigmoid')) #Capa de salida

# Compilar el modelo
# Parametros (optimizador = algoritmo de descenso de gradiente estocastico, perdida = funcion de perdida, metrica = metrica de rendimiento)
clasificador.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
#clasificador.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('training_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

clasificador.fit_generator(training_set,
                            steps_per_epoch = 8000,
                            epochs = 1,
                            validation_data = test_set,
                            validation_steps = 2100)

training_set.class_indices
# Guardar el modelo creado
model_json = clasificador.to_json()
with open("animals.json", "w") as json_file:
    json_file.write(model_json)
# serializar los pesos a HDF5
clasificador.save("animalsPesos.h5")
print("Modelo Guardado!")