from keras.models import Sequential, model_from_json
from keras.layers import Dense
import numpy
# Fija las semillas aleatorias para la reproducibilidad
numpy.random.seed(7)

# carga los datos
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# dividido en variables de entrada (X) y salida (Y)
X = dataset[:,0:8]
Y = dataset[:,8]
#print(Y)

# crea el modelo
model = Sequential()
model.add(Dense(24, input_dim=8, activation='relu')) # Capa Oculta
model.add(Dense(8, activation='relu')) # Capa de entrada
model.add(Dense(1, activation='sigmoid')) # Capa de salida

# Compila el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # binary_crossentropy: Perdida logaritmica, adam: Descenso de gradiente eficiente

# Ajusta el modelo
model.fit(X, Y, epochs=500, batch_size=10) # epochs: Epocas(numero de iteraciones), batch_size: numero de instancias que se evaluan antes de que se realice una actualizacion de peso

# evalua el modelo
scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calcula las predicciones
predictions = model.predict(X)
# redondeamos las predicciones
rounded = [round(x[0]) for x in predictions]
#print(rounded)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serializar los pesos a HDF5
model.save_weights("model.h5")
print("Modelo Guardado!")
 
# mas tarde...
"""
# cargar json y crear el modelo
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# cargar pesos al nuevo modelo
loaded_model.load_weights("model.h5")
print("Cargado modelo desde disco.")
 
# Compilar modelo cargado y listo para usar.
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
"""