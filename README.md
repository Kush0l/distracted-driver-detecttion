# Distracted Driver Detection


This project aims to detect drowsiness and mobile phone usage by drivers in real-time using computer vision and deep learning techniques. The workflow starts by initializing necessary libraries and loading the pre-trained YOLO model for object detection, along with dlib's face detector and shape predictor for facial landmark detection. The video feed is captured from the webcam, and each frame is processed to detect faces and eyes, with the eye aspect ratio (EAR) calculated to monitor drowsiness. Simultaneously, the YOLO model identifies objects in the frame, specifically targeting mobile phones. If a mobile phone is detected near a driver's face or the EAR indicates drowsiness, visual alerts are displayed on the frame, and an auditory alarm is triggered for drowsiness. The program continuously processes the video feed, updating detections and alerts in real-time, until the user stops the program.

## installation process
1. clone the repo  
`git clone https://github.com/Kush0l/distracted-driver-detecttion.git`

2. install packages in requerments.txt
`pip install -r requirements.txt`

3. install yolov3.weights from here here [https://www.kaggle.com/datasets/shivam316/yolov3-weights]

4. run the main.py
`python main.py`
