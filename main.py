from model import CordModel
import cv2
import mediapipe as mp
from PIL import Image
import torch
import torch.nn 
import numpy as np
from flask import Flask,Response,render_template
app = Flask(__name__)

model = CordModel()
model.load_state_dict(torch.load('my_new_model.pth',map_location=torch.device('cpu')))
model.eval()


cap=cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hand=mp_hands.Hands()



def coord_to_tensor(landmarks):
    cord=np.array([])
    for i in range (21):
       cord=np.append(cord,landmarks[i].x)
       cord=np.append(cord,landmarks[i].y)
       cord=np.append(cord,landmarks[i].z)
    crd=torch.tensor(cord,dtype=torch.float)
    return crd 
  
num_class=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
def gen_frame():
 while True:
   suc,frame=cap.read()
   if suc:
      frame_RGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      results=hand.process(frame_RGB)
      if results.multi_hand_landmarks:
          for data in results.multi_hand_landmarks:
           crd=coord_to_tensor(data.landmark)
           crd=crd.reshape(1,63)
           pred=model(crd)
           txt=str(num_class[pred.argmax(1)])
           cv2.putText(
                    frame,  txt,  (100, 100),  
                    cv2.FONT_HERSHEY_SIMPLEX, 1,  
                    (0, 255, 255),  
                    2,  
                    cv2.LINE_4)
           print(txt)
           mp_drawing.draw_landmarks(frame,data,mp_hands.HAND_CONNECTIONS)
      else:
          print("NO hand detected !!")
      cv2.waitKey(1)
      cv2.imshow("Frame",frame)
      ret, buffer = cv2.imencode('.jpg', frame)
      frame = buffer.tobytes()
      yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
  
if __name__ == "__main__":
    app.run(debug=True)