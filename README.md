# Sign Language Translator

This is a python project which will translate American Sign Language in to English Text by reading keras model trained from Teachable Machine. 

Illness like deaf and dumb are never a rare occurrence in the current society no matter it is natural or acquired. They face difficulty to interact and communicate with the others and a decent solution is hard to come by on the market. Hence, this problem has inspired the author to develop an application to resolve this issue. 

The application is called as Sign Language Translator, and it is used to recognize and translate America Sign Language to text. The project comes with both Web Based Application and a Python Application that have certain difference in functionality and will be discussed in the following documents. This project aims to extend a step forward in this field by carry out a series of research in image processing and machine learning to provides a functional solution for the deaf mute to perform basic communication in daily life. 

The application takes sign language as input and returns the corresponding recognized letter as output in real time on a monitor screen. It can recognize up to 26 letters from letter A to Z. The model is trained using Teachable Machine powered by Google that applied Convolutional Neural Network Transfer Learning for the sign language classification. 

Communication is always two-way. The deaf-mute learns how to write, recognizing the words, and learn lip reading to integrate into the world of normal people; hence the normal people should also make a little effort to understand and accept the deaf mute. To this day, sign language is still the most natural and efficient way of expression for many deaf mute people, it is faster than writing and typing. However, in order to promote barrier-free communication, it is obviously unrealistic to require all normal people to learn sign language. Thus, the author strongly believe that the end product of the project will be developed successfully and will bring an excellent convenience to the deaf mute, as well as creating a new paradigm in deaf mute communication. 

## Getting Started
Please install the required libraries from requirements.txt before running the project.
> pip install -r /path/to/requirements.txt

## Running The Project
1. Download the projects.
2. Make sure all the files are under the same folder.
3. Run the project in VS Code or Manually by setting the following command in the command prompt. 
   >FLASK_APP = main.py
4. Run the program.

## Keras Model
You can train your own keras model and replace it in the project by changing
`model = keras.models.load_model('your own model',compile=False).`
Model can be trained in [Teachable Machine](https://teachablemachine.withgoogle.com/)

## Project Preview
![image](https://user-images.githubusercontent.com/102412283/164979138-0faa0ed8-6162-4154-a8c5-c98d7c4aed65.png)

