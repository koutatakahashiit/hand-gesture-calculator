import cv2

device = 0

cap = cv2.VideoCapture(device)
if not cap.isOpened():
    print("camera open error")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("frame read error")
        break

    cv2.imshow("Camera", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()