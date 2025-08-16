import face_recognition as face 
import numpy as np 
import cv2

#ORIGINAL_CODE_CREDIT:  https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
'''
ต้องโหลดอะไรก่อนใช้โปรแกรม
- File
    โหลด https://github.com/justra1/Detect-face-my-friend/blob/main/dlib-20.0.0-cp313-cp313-win_amd64.whl เอาไว้ใน drive เดี๋ยวกับไฟล์ code
- Languge
    python 3.13.x (ver. นี้เท่านั้น)
- Extensions in vs code
    Python
    Pylance
    Python Debugger
    Python Environment
- Command prompt
    python -m venv myenv
    pip install opencv-python
    pip install cmake
    pip install "อันนี้ต้องใส่ที่อยู่ (path) ของ dlib-20.0.0-cp313-cp313-win_amd64.whl ที่อยู่ใน drive ของเรา" 
        ex. pip install "D:\dlib-20.0.0-cp313-cp313-win_amd64.whl"
    pip install face-recognition
'''
video_capture = cv2.VideoCapture(0) 

if not video_capture.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

# reference face
Your_image = face.load_image_file("image path")
Your_encoding = face.face_encodings(Your_image)[0]


known_face_encodings = [Your_image]
known_face_names = ["Class name"]

face_locations = []
face_encodings = []
face_names = []
face_percent = []
process_this_frame = True 

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # ลด scale เหลือ 1/4 เพื่อประหยัดพลังงาน
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1].astype(np.uint8)

    face_names = []
    face_percent = []

    if process_this_frame:
     
        face_locations = face.face_locations(rgb_small_frame, model="hog") #โค้ดนี้แปลง model จาก cnn เป็น hog
        face_encodings = face.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            face_distances = face.face_distance(known_face_encodings, face_encoding)
            best = np.argmin(face_distances)
            face_percent_value = 1-face_distances[best]
            
            if face_percent_value >= 0.5: #ปรับ percent 
                name = known_face_names[best]
                percent = round(face_percent_value*100,2)
                face_percent.append(percent)
            else:
                name = "UNKNOWN"
                face_percent.append(0)
            face_names.append(name)

    for (top,right,bottom,left), name, percent in zip(face_locations, face_names, face_percent):
        # คูณกลับ 4 เพราะย่อ 1/4
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name == "UNKNOWN":
            color = [46,2,209]
        else:
            color = [255,102,51]

        cv2.rectangle(frame, (left,top), (right,bottom), color, 2)
        cv2.rectangle(frame, (left-1, top -30), (right+1,top), color, cv2.FILLED)
        cv2.rectangle(frame, (left-1, bottom), (right+1,bottom+30), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, top-6), font, 0.6, (255,255,255), 1)
        cv2.putText(frame, "MATCH: "+str(percent)+"%", (left+6, bottom+23), font, 0.6, (255,255,255), 1)

    

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
