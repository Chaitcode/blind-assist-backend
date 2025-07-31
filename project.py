import speech-recognition-python as sr
import pyttsx3
import opencv-python as cv2
import numpy as np
import wikipedia
import pyjokes
import pywhatkit
import datetime
import matplotlib.pyplot as plt
import os
import pyaudio


config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

model=cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name = 'labels.txt'
with open(file_name,'rt')as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

   

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127,5,127.5))
model.setInputSwapRB(True)

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
def talk(text):
    engine.say(text)
    engine.runAndWait()

import speech-recognition-python as sr

def take_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.pause_threshold = 1
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            command = ""  # Assign an empty string if recognition fails
        except sr.RequestError:
            print("Could not request results, check your internet connection.")
            command = ""  # Assign an empty string in case of API failure
    return command  # Ensure command is always defined



 #webcam    
def webcam():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    font_scale =3
    font =cv2.FONT_HERSHEY_PLAIN

    while True:
        ret, frame = cap.read()

        ClassIndex, confidece, bbox = model.detect(frame, confThreshold = 0.55)
        print(ClassIndex)

        if(len(ClassIndex)!=0):
            for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
                if(ClassInd<=80):
                    cv2.rectangle(frame,boxes,(255,0,0),1)
                    object = cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=1)
                    print(object)
        cv2.imshow('object detection',frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        answer= classLabels[ClassInd-1]
        talk(answer)

    cap.release()
    cv2.destroyAllWindows()


def run_alexa():
    command=take_command()
    if 'play' in command:
        song=command.replace('play','')
        print("playing")
        talk("playing"+song)
        pywhatkit.playonyt(song)
    elif 'time' in command:
        time=datetime.datetime.now().strftime('%I:%M %p')
        talk('current time is '+time)
        print(time)
    elif 'tell me about' in command:
        person= command.replace('tell me about','')
        info = wikipedia.summary(person,1)
        print(info)
        talk(info)
    elif 'joke' in command:
        talk(pyjokes.get_joke())
    elif 'detect object' in command:
        webcam()

run_alexa()
   

   

