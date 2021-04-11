

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

cnn=load_model('modelo.h5')
cnn.load_weights('pesos.h5')

x=load_img('examples/golf.jpg', target_size=(21, 28))
x=img_to_array(x)/255
x=np.expand_dims(x, axis=0)
arreglo=cnn.predict(x)
resultado=arreglo[0]
respuesta=np.argmax(resultado)
percentage=resultado[respuesta]*100
print(percentage)


if respuesta==0:
    print('tenis')
if respuesta==1:
    print('americano')
if respuesta==2:
    print('f1')
if respuesta==3:
    print('natacion')
if respuesta==4:
    print('futbol')
if respuesta==5:
    print('ciclismo')
if respuesta==6:
    print('golf')
if respuesta==7:
    print('basket')
if respuesta==8:
    print('boxeo')
if respuesta==9:
    print('beisball')