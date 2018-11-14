import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import ImgKeras as ik
from keras.utils import to_categorical
 
print('Training data shape : ', ik.train_images.shape, ik.train_labels.shape)
 
print('Testing data shape : ', ik.test_images.shape, ik.test_labels.shape)
 
# Find the unique numbers from the train labels
classes = np.unique(ik.train_labels)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
 
plt.figure(figsize=[10,5])
 
# Display the first image in training data
plt.subplot(121)
plt.imshow(ik.train_images[0,:,:], cmap='gray')
plt.title("Muestra verdadera : {}".format(ik.train_labels[0]))
 
# Display the first image in testing data
plt.subplot(122)
plt.imshow(ik.test_images[0,:,:], cmap='gray')
plt.title("Muesta verdadera : {}".format(ik.test_labels[0]))
plt.show()

# Change from matrix to array of dimension 28x28 to array of dimention 784
dimData = np.prod(ik.train_images.shape[1:])
train_data = ik.train_images.reshape(ik.train_images.shape[0], dimData)
test_data = ik.test_images.reshape(ik.test_images.shape[0], dimData)

# Change to float datatype
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
 
# Scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255

# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(ik.train_labels)
test_labels_one_hot = to_categorical(ik.test_labels)
 
# Display the change for category label using one-hot encoding
print('Original label 0 : ', ik.train_labels[0])
print('Despues de la conversion a categorico ( one-hot ) : ', train_labels_one_hot[0])