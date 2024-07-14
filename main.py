import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer

from ear import eye_aspect_ratio


# Initialize mixer for alert sound
mixer.init()
mixer.music.load('alert.wav')





# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')



# Thresholds and frame check for drowsiness detection
thresh = 0.25
frame_check = 7
flag = 0
face_frame_check = 5

# Initialize the video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, channels = frame.shape

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = face_detector(gray, 0)

# face destraction
    if len(subjects) == 0:
        face_flag += 1
        if face_flag > face_frame_check:
            # Print error message when no face is detected
            print("No face detected!")
            cv2.putText(frame, "****************No face detected!!****************", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "****************No face detected!!****************", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "No face detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        face_flag = 0 
    
    # Object detection (mobile phones)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    

    for subject in subjects:
            
            # Get the landmarks/parts for the face in the bounding box
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

             # Extract left and right eye coordinates
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw the contours around eyes
            # get convex shape of eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            # draw around convex
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


            # check thresh 
            if ear < thresh:
                flag += 1
                print(flag)
                if flag >= frame_check:
                    cv2.putText(frame, "****************Drowsy!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************Drowsy!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    mixer.music.play()
                    print("Drowsy")
            else:
                flag = 0

            # Draw the face bounding box
            (x, y, w, h) = (subject.left(), subject.top(), subject.width(), subject.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)    

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == classes.index("cell phone"):
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    

    for i in range(len(boxes)):
        if i in indexes:
            box_x, box_y, box_w, box_h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
            cv2.putText(frame, label, (box_x, box_y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        # Check if the mobile phone is near the detected face
            if (x < box_x < x + w or x < box_x + box_w < x + w) and (y < box_y < y + h or y < box_y + box_h < y + h):
                cv2.putText(frame, "Using Mobile Phone", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
