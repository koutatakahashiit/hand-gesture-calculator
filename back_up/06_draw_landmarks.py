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
    
    if result.hand_landmarks: #手が検出された場合
        message = "Hand detected"
        height, width, _ = frame.shape #画像の高さ，幅，色．今回色情報は使わない

        for hand_landmarks in result.hand_landmarks: #検出された手を1つずつ取り出す(今回の場合，一人なので最大2つ)
            for landmark in hand_landmarks: #手の21個の点を取り出す．
                
                #landmarkは端から見た座標の比率(0.0-1.0)で表される
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame, #描画対象画像
                    (x,y), #描画位置(円の中心)
                    4, #半径
                    (0,255,255), #色(黄色)
                    -1 #塗りつぶし(塗りつぶさないため，-1)
                )
                
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