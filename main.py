from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
from rsa import sign
import tensorflow.keras as keras
from keras.models import load_model
import sys

app = Flask(__name__)
camera  = cv2.VideoCapture(0)
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

        

def cameraOnOff():
    if camera is None or not camera.isOpened():
        return True
    else:
        return False

def generate_frames():
    while True:
        if camera.read()[0]==False:
            return
        else:
            success, frame = camera.read() 
            frame = image_resize(frame,height=224)
            frame = cropTo(frame)

            frame = cv2.flip(frame,1)

            #normalize the frame and laod it into array to fit the format for keras
            normalized_frame = (frame.astype(np.float32)/127.0)-1
            data_for_model[0] = normalized_frame

            prediction = model.predict(data_for_model)
            move_code = np.argmax(prediction[0])            
            move_name = mapper(move_code)
            cv2.putText(frame, "Sign: "+move_name,(100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            
            if not success:
                break
            else:
                ret, buffer= cv2.imencode('.jpg',frame)
                
                

                frame=buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route("/")
def index(condition = False):
        status=""
        if camera.read()[0]==False:
            status = "Camera is in used by other application."
        else:
            status=""
        return render_template('index.html',condition=cameraOnOff(),status=status)


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')



if __name__=='__main__':
    app.run(debug=False)
