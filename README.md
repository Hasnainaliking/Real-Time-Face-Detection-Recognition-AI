# Real-Time Face Detection & Recognition using AI
This project is a practical implementation of real-time face detection and recognition using AI techniques powered by OpenCV and the LBPH (Local Binary Pattern Histogram) algorithm.

# Features
Captures face images via webcam

Trains a face recognizer model on collected data

Detects and recognizes faces in real-time

Displays recognition confidence score live

# Technologies Used:
Python

OpenCV (opencv-contrib-python)

NumPy

PIL

Haar Cascade Classifier

LBPH Face Recognizer

# project Structure
face_collection.py: Collects face images and saves them to a dataset

train.py: Trains the model using collected face images

detect.py: Performs real-time face recognition

# How It Works
User provides their name and ID

System captures 100 face samples

Model is trained on the dataset

Real-time face detection and recognition is done via webcam

# How to Run
Install dependencies:

bash
pip install opencv-contrib-python numpy pillow

# Run face_collection.py to gather face samples

# Run train.py to train the model

# Run detect.py to perform real-time recognition
