import Prueba as p
from time import time 
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
 
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(p.dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(p.nClasses, activation='softmax'))

"""
# Para reajustar el sobreajuste (parámetros de red se están inclinando demasiado hacia los datos de entrenamiento)
from keras.layers import Dropout
 
model_reg = Sequential()
model_reg.add(Dense(512, activation='relu', input_shape=(dimData,)))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(512, activation='relu'))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(nClasses, activation='softmax'))
"""

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1, write_images=True)

history = model.fit(p.train_data, p.train_labels_one_hot, batch_size=256, epochs=20, verbose=1, 
                    validation_data=(p.test_data, p.test_labels_one_hot), callbacks=[tensorboard])

[test_loss, test_acc] = model.evaluate(p.test_data, p.test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Perdida de entrenamiento', 'Perdida de validacion'],fontsize=18)
plt.xlabel('Epocas ',fontsize=16)
plt.ylabel('Perdida',fontsize=16)
plt.title('Curva de Pedida',fontsize=16)
plt.show()
 
#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Precision de Entrenamiento', 'Precision de Validacion'],fontsize=18)
plt.xlabel('Epocas ',fontsize=16)
plt.ylabel('Precision',fontsize=16)
plt.title('Curva de Precision',fontsize=16)
plt.show()