from scipy.spatial import distance
import cv2
import dlib
from imutils import face_utils
from pygame import mixer

mixer.init()
mixer.music.load('alert.wav')

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

# video capture from the webcam
cap = cv2.VideoCapture(0)


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

thresh = 0.25 # threshold for EAR
flag = 0  #no. of frames
frame_check = 25 #maximum threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # converts frame to gery scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces 
    subjects = detector(gray, 0)
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
    
    # Display the output
    cv2.imshow("Infothon", frame) 

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()