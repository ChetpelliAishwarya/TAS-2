import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.utils import to_categorical
import tkinter as tk
from PIL import Image, ImageTk

model = load_model(r'C:\Users\chais\Desktop\TASK-2\Sign detection model\signlanguagedetectionmodel.h5')

image_height, image_width = 64, 64
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del']

phrase_map = {
    'WHATISYOURNAME': 'Who are you',
    'WHOAREYOU': 'What is your name'
}

def preprocess_frame(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_gray, (image_height, image_width))
    frame_resized = np.expand_dims(frame_resized, axis=0)
    frame_resized = np.expand_dims(frame_resized, axis=-1)
    frame_resized = frame_resized.astype('float32') / 255.0
    return frame_resized

def predict(frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

def map_phrases(predicted_sentence):
    joined_sentence = ''.join(predicted_sentence).replace(' ', '')
    for key in phrase_map:
        if key in joined_sentence:
            return phrase_map[key]
    return ''.join(predicted_sentence)

cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title('ASL Alphabet Prediction')
prediction_label = tk.Label(root, text='Predicted Sentence:')
prediction_label.pack()

sentence_label = tk.Label(root, text='')
sentence_label.pack()

frame_label = tk.Label(root)
frame_label.pack()

def update_sentence():
    global predicted_sentence
    mapped_sentence = map_phrases(predicted_sentence)
    sentence_label.config(text=mapped_sentence)
    root.after(100, update_sentence)

predicted_sentence = []

import time
last_prediction_time = time.time()

def video_loop():
    global predicted_sentence, last_predicted_sign, last_prediction_time

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        frame_label.imgtk = imgtk
        frame_label.config(image=imgtk)

        current_time = time.time()
        if current_time - last_prediction_time >= 1.0:  # Adjust the time interval as needed
            predicted_sign = predict(frame)

            if predicted_sign != last_predicted_sign:
                if predicted_sign == 'space':
                    predicted_sentence.append(' ')
                elif predicted_sign == 'del':
                    if predicted_sentence:
                        predicted_sentence.pop()
                else:
                    predicted_sentence.append(predicted_sign)

                last_predicted_sign = predicted_sign
                last_prediction_time = current_time

            mapped_sentence = map_phrases(predicted_sentence)
            sentence_label.config(text=mapped_sentence)

    root.after(10, video_loop)

root.after(10, video_loop)
root.after(100, update_sentence)
root.mainloop()
