import numpy as np

# X = (horas durmiendo, horas estudiando), y = puntuacion en la prueba
X = np.array(([2, 8, 9], [1, 8, 5], [3, 8, 6], [10, 8, 2], [14, 8, 1]), dtype=float)
y = np.array(([92], [75], [89], [40], [20]), dtype=float)
print(X)
# unidades de escala
X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # puntaje maximo de la prueba es de 100

class Neural_Network(object):
    def __init__(self):
        #parametros
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 8

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

    def forward(self, X):
        #Propagacion hacia adelante a traves de nuestra red
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        return o

    def sigmoid(self, s):
        # Funcion de activacion 
        return 1/(1+np.exp(-s)) 
    
    def sigmoidPrime(self, s):
        # Derivada de Sigmoid
        return s * (1 - s)
    
    def backward(self, X, y, o):
        # propagación hacia atras a traves de la red
        self.o_error = y - o # Error en la salida
        self.o_delta = self.o_error*self.sigmoidPrime(o) # Aplicando derivado de sigmoide al error.

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: Cuanto contribuyeron nuestros pesos de la capa oculto al error de salida
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # aplicando la derivada de sigmoide a z2 error

        self.W1 += X.T.dot(self.z2_delta) # Ajuste de los pesos del primer conjunto (entrada -> ocultos)
        self.W2 += self.z2.T.dot(self.o_delta) # ajuste de los pesos del segundo conjunto (oculto -> salida)

    def train (self, X, y):
        # Para iniciar con la funcion propagar(forward) y luego continuar con la funcion backward
        o = self.forward(X)
        self.backward(X, y, o)

NN = Neural_Network()

#Definimos nuestra salida 
for i in range(1000):
    o = NN.forward(X)
    #print ("Entrada: \n" + str(X))
    #print ("Salida predecida: \n" + str(o))
    #print ("Salida actual: \n" + str(y))
    print ("Perdida: \n" + str(np.mean(np.square(y - o)))) #sum media de perdida al cuadrado

    NN.train(X, y)

print("BIENVENIDO a CASA")
h = NN.forward(np.array(([11, 8, 1]), dtype=float))
print ("Entrada: \n" + str(np.array(([11, 8, 1]), dtype=float)))
print ("Salida predecida: \n" + str(h))