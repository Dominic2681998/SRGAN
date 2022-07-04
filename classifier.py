import tensorflow as tf
from tensorflow import keras
import numpy as np
#from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model
#import cv2
import matplotlib.pyplot as plt


from torch import randint


def classify(imgname):
    
    generator = load_model('model/gen_e_12.h5', compile=False)
    image =load_img("static/uploads/"+imgname,target_size=(128,128))
    input_arr=img_to_array(image)/255#dividing the image by 255 to normalise the image
    input_arr=np.expand_dims(input_arr,axis=0)
    pred=generator.predict(input_arr)
    gan=np.reshape(pred, (512,512, 3))
    gan=gan/255
   
    tf.keras.preprocessing.image.save_img(f'result/'+imgname, gan)
    #cv2.imwrite("result/1.jpg", gan)
    return True

#def classify(imgname):

    #image = read_image("static/uploads/"+imgname)
    

   # p = predict_x(x)
    
    #return p


