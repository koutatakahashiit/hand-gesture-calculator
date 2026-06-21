import cv2
import mediapipe as mp

device = 0
model_path = "models/hand_landmarker.task" #手検出モデルの場所

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions( #手検出器の設定
    base_options=BaseOptions(model_asset_path=model_path), #モデルファイル
    running_mode=VisionRunningMode.IMAGE, #1枚ずつ処理する
    num_hands=2, #最大の手の検出数
    min_hand_detection_confidence=0.5,#手の検出信頼度
    min_hand_presence_confidence=0.5, #手ランドマークに対しての手の検出信頼度
)

detector = HandLandmarker.create_from_options(options) #検出器の作成

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
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #MediaPipe用の形式(RGB画像)にするために，CV2のBGR形式からRGBに変換．
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )
    
    result = detector.detect(mp_image) #手の検出をする処理
    
    if result.hand_landmarks: #手が検出されたか
        message = "Hand detected"
    else:
        message = "No hand"
    
    cv2.putText(
        frame, #対象画像
        message, #表示する文字
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