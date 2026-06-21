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

    frame = cv2.flip(frame, 1) #画面反転．引数1が左右反転
    
    cv2.putText(
        frame, #対象画像
        "Press q to quit", #表示する文字
        (20, 40), #文字の位置．座標は文字の左下当たりの位置を指す
        cv2.FONT_HERSHEY_SIMPLEX, #フォント
        1.0, #文字サイズ
        (0, 255, 0), #色
        2 #太さ
    )
    
    cv2.imshow("Camera", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()