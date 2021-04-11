

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

cnn=load_model('modelo.h5')
cnn.load_weights('pesos.h5')

x=load_img('examples/futball.jpg', target_size=(21, 28))
x=img_to_array(x)
x=np.expand_dims(x, axis=0)
arreglo=cnn.predict(x)
resultado=arreglo[0]
respuesta=np.argmax(resultado)
print()
if respuesta==0:
    print('tenis')
