from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp 
import cv2 
import pickle
import pandas as pd
from gtts import gTTS
import os
import pyttsx3
from playsound import playsound
app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
holistic = mp_holistic.Holistic()



mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/generate_frames', methods=['GET', 'POST'])
def generate_frames():
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
        while cap.isOpened():
            ret, frame = cap.read()
            
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            
            results = holistic.process(image)
            
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            

            
            
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                     )

            
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                     )


            print('sad')
            
            try:
                # Extract Pose landmarks
                pose = results.left_hand_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                
                print('Happy')
                
                row = pose_row
                
                  
                print('victorious')
                 

                # Make Detections
                X = pd.DataFrame([row])
                print('X')
                body_language_class = model.predict(X)[0]
                print('Output;',body_language_class)
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_prob)
                print(body_language_class, body_language_prob)
                speak=gTTS(text=body_language_class)
                speak.save("show.mp3")
                playsound('show.mp3')
                os.remove('show.mp3')
                
                # Grab ear coords
                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                cv2.rectangle(image, 
                              (coords[0], coords[1]+5), 
                              (coords[0]+len(body_language_class)*20, coords[1]-30), 
                              (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (0, 0, 0), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                

            
            except Exception as e:
               print(e)
            except:
                pass
                            
            cv2.imshow('Camera Feed with Predictions', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                
                break

    cap.release()
    cv2.destroyAllWindows()
    return render_template("index.html")
  
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=800)
