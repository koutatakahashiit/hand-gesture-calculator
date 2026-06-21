import cv2
import mediapipe as mp

import numpy as np

device = 0 #カメラのデバイス番号．内部カメラを用いるので0
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
connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS #手のランドマーク同士の接続情報

# 映像取得
cap = cv2.VideoCapture(device)
if not cap.isOpened():
    print("camera open error")
    exit()

#二点間の距離を計算する関数(landmarkのx,yを用いる)
def calc_dist2d(p1,p2):
    p1_xy = np.array([p1.x, p1.y])
    p2_xy = np.array([p2.x, p2.y])
    return float(np.linalg.norm(p1_xy - p2_xy)) #ユークリッド距離(直線距離)

#指の本数を数える関数(改良の余地あり．距離計算のみのため)
# 試してみたいもの：ランドマークを入力とする深層学習
def count_fingers(hand_landmarks):
    wrist = hand_landmarks[0] #手首
    bias = 1.15 #パラメータ

    # 指のランドマーク番号
    finger_pairs = [
        (4,2), #親指．4が指先，2が関節
        (8,6), #人差し指．8が指先，6が関節
        (12,10), #中指．12が指先，10が関節
        (16,14), #薬指．16が指先，14が関節
        (20,18) #小指．20が指先，18が関節
    ]
    
    cnt = 0
    
    for tip_index, base_index in finger_pairs:
        tip = hand_landmarks[tip_index]
        base = hand_landmarks[base_index]
        
        #手首から指先までの距離と，手首から関節までの距離をパラメータ(bias)で比較し，大きい場合立てたと判定し，数を加える
        if calc_dist2d(wrist,tip) > calc_dist2d(wrist, base) * bias:
            cnt += 1
    return cnt


while True:
    # 各フレームごと，つまり画像を取得
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
    
    # 手の検知を行う
    result = detector.detect(mp_image) #手の検出をする処理
    if result.hand_landmarks: #手が検出された場合
        height, width, _ = frame.shape #画像の高さ，幅，色．今回色情報は使わない

        # 検出した両手の描画処理
        for hand_landmarks in result.hand_landmarks: #検出された手を1つずつ取り出す(今回の場合，一人なので最大2つ)
            message = f"Fingers {count_fingers(hand_landmarks)}"

            # 手の線を描画する
            for connection in connections: #1つずつ接続情報(始点と終点)を取り出す
                start_index = connection.start #始点の添え字
                end_index = connection.end #終点の添え字

                start = hand_landmarks[start_index] #始点座標の比率(0.0-1.0)を取得
                end = hand_landmarks[end_index] #終点座標の比率(0.0-1.0)を取得

                x1 = int(start.x * width) #x座標の始点
                y1 = int(start.y * height) #y座標の始点

                x2 = int(end.x * width) #x座標の終点
                y2 = int(end.y * height) #y座標の終点

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

            # 手の点を描画する
            for i, landmark in enumerate(hand_landmarks): #手の21個の点を取り出す．

                #landmarkは端から見た座標の比率(0.0-1.0)で表される
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                #指先だけ赤く，他は黄色にする(指先のランドマークが4, 8, 12, 16, 20のため)
                if i != 0 and i % 4 == 0:
                    color = (0,0,255)
                else:
                    color = (0,255,255)

                cv2.circle(
                    frame, #描画対象画像
                    (x,y), #描画位置(円の中心)
                    4, #半径
                    color, #色
                    -1 #塗りつぶし
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