import cv2 # import opencv
import tensorflow.keras as keras
import numpy as np
import time
from keras.models import load_model
from tensorflow.keras import layers
from PIL import Image, ImageOps
import tkinter
from tkinter import messagebox




text = ""
camera = cv2.VideoCapture(0)
model = keras.models.load_model('keras_model.h5',compile=False)
data_for_model = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

REV_CLASS_MAP ={0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G",7:"H",8:"I",9:"J",
                10:"K",11:"L",12:"M",13:"N",14:"O",15:"P",16:"Q",17:"R",18:"S",
                19:"T",20:"U",21:"V",22:"W",23:"X",24:"Y",25:"Z"}
def mapper(val):
    return REV_CLASS_MAP[val]

# This function proportionally resizes the image from your webcam to 224 pixels high
def image_resize(image, height, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

# this function crops to the center of the resize image
def cropTo(img):
    size = 224
    height, width = img.shape[:2]

    sideCrop = (width - 224) // 2
    return img[:,sideCrop:(width - sideCrop)]
    

if camera is None or not camera.isOpened():
    messagebox.showerror('Error','Camera not found!')
else:
    while True:
        if camera.read()[0]==False:
             messagebox.showerror('Error','Camera is in used!')
             break
        else:
            ret, img = camera.read()
            if ret:
                textFrame = np.full((224,224,3),(0,0,0),np.uint8)
                #same as the cropping process in TM2
                img = image_resize(img, height=224)
                img = cropTo(img)

                # flips the image
                img = cv2.flip(img, 1)

                #normalize the image and load it into an array that is the right format for keras
                normalized_img = (img.astype(np.float32) / 127.0) - 1
                data_for_model[0] = normalized_img
        
                #run inference
                prediction = model.predict(data_for_model)
                move_code = np.argmax(prediction[0])
                move_name = mapper(move_code)
                
                
                accuracy = prediction[0][move_code]*100
                cv2.putText(img, "Sign: " + move_name + " "+ str(accuracy),(0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                y0, dy = 0,50
                for i, newText in enumerate(text.split('\n')):
                    y = y0 + i*dy
                    cv2.putText(textFrame, newText,(0,y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  
                    
                # concatanate image Horizontally
                Hori = np.concatenate((img, textFrame), axis=1)
        
                
                cv2.imshow('Sign Language Translator Application', Hori)
                keypress = cv2.waitKey(1)
                if keypress == ord('v'):
                    text = text + move_name
                if keypress == ord('c'):
                    text = ""
                if keypress == ord('n'):
                    text = text + "\n"
                if keypress == 8:
                    text = text[:-1]
                if keypress == 32:
                    text = text + " "
                if keypress == 27:
                    break


                   
camera.release()
cv2.destroyAllWindows()