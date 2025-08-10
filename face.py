import cv2
from ultralytics import YOLO

# โหลดโมเดลจากไฟล์ weight
model = YOLO("weight face.pt")

# เปิด webcam (0 = กล้องหลัก)
cap = cv2.VideoCapture(0)
#---------------------------------- อ้างอิงคลิปนี้ครับ https://www.youtube.com/watch?v=6RuaEI1dP_Q&t=902s
while True:
    ret, frame = cap.read()
    if not ret:
        break
#----------------------------------
    # ตรวจจับจาก frame
    results = model(frame, stream=True)

    for r in results:
        # ใช้ plot() เพื่อให้ YOLO วาดกรอบ + label ให้เลย (ผมถามข้อมูล chatgpt ครับ แต่ผมเข้าใจคือมันจะวาดกรอบที่เป็น weight ไฟล์จาก detect หรือ segment มันจะวาดและตั้งชื่อตาม classes ให้เลย)
        annotated_frame = r.plot()
#----------------------------------- ล่างผมอ้างอิงจากคลิปนี้ครับ https://www.youtube.com/watch?v=ine6kW1AaOA
        # แสดงผล
        cv2.imshow("Detect my friend", annotated_frame)

    # ออกจากโปรแกรมเมื่อกด 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
