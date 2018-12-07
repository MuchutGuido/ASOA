from keras.applications.inception_resnet_v2 import InceptionResNetV2             
from keras.preprocessing import image                                            
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import numpy as np
from goslate import Goslate
from time import time 

start_time = time()     
gs = Goslate()                                                                     
model = InceptionResNetV2(weights='imagenet')                                    
                                                                    
img_path = 'ImgPrueba/perro2.jpg'                                                          
img = image.load_img(img_path, target_size=(224, 224))                           
x = image.img_to_array(img)                                                      
x = np.expand_dims(x, axis=0)                                                    
x = preprocess_input(x)                                                          
                                                                                 
preds = model.predict(x)
#preds('Prueba: ', preds[0][0])
predi = decode_predictions(preds, top=1)[0][0]
nomAnimal = predi[1]
porc = predi[2]

#print ('Predicion:', decode_predictions(preds, top=1)[0][0])
print('Nombre del Animal: ', nomAnimal)#gs.translate(nomAnimal, 'es'))
print('Probabilidad: ', porc)
elapsed_time = time() - start_time
print("Elapsed time: %.10f seconds." % elapsed_time)