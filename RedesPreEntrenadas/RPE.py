from keras.applications.inception_resnet_v2 import InceptionResNetV2             
from keras.preprocessing import image                                            
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import numpy as np                                                               
                                                                                 
model = InceptionResNetV2(weights='imagenet')                                    
                                                                    
img_path = 'ImgPrueba/aguila.jpeg'                                                          
img = image.load_img(img_path, target_size=(224, 224))                           
x = image.img_to_array(img)                                                      
x = np.expand_dims(x, axis=0)                                                    
x = preprocess_input(x)                                                          
                                                                                 
preds = model.predict(x)
#preds('Prueba: ', preds[0][0])                                                        
print ('Predicion:', decode_predictions(preds, top=1)[0][0])