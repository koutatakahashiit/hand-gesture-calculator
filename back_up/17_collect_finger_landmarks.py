import cv2 #カメラ，画像表示．描画処理に用いる
import mediapipe as mp #手の検出・ランドマーク取得に用いる
import numpy as np #2点間距離の計算に用いる

import csv #学習データをCSVに保存するために用いる
import os #保存先フォルダの作成とファイル存在確認に用いる

DEVICE = 0 #使用するカメラ番号．通常は最初に認識されたカメラが0になる

CSV_PATH = "data/hand_landmarks_and_states.csv" #学習データの保存先
MODEL_PATH = "models/hand_landmarker.task" #手検出モデルファイルのパス

FINGER_TIP_INDICES = [4, 8, 12, 16, 20] #指先のランドマーク番号

#簡易化
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

#二点間の距離を計算する関数(landmarkのx,yを用いる)
def calc_dist2d(p1,p2):
    p1_xy = np.array([p1.x, p1.y])
    p2_xy = np.array([p2.x, p2.y])
    return float(np.linalg.norm(p1_xy - p2_xy)) #ユークリッド距離(直線距離)

#ランドマークを機械学習用の特徴量に変換する関数
def landmarks_to_features(hand_landmarks):
    wrist = hand_landmarks[0] #手首

    #手首から中指の付け根までの距離を手の大きさとする．(手の動作による影響を受けにくい部位のため)
    scale = calc_dist2d(hand_landmarks[0], hand_landmarks[9])

    #0除算を避ける
    if scale == 0:
        return None

    features = []
    #21点ランドマークを1点(x,y,z)ずつ正規化した後，特徴量に追加する
    for landmark in hand_landmarks:
        #手首を基準とした相対座標にし，手の大きさで割ることで正規化する
        x = (landmark.x - wrist.x) / scale
        y = (landmark.y - wrist.y) / scale
        z = (landmark.z - wrist.z) / scale
        features.extend([x, y, z])

    return features

#左右反転した映像に合わせて，MediaPipeの左右判定を入れ替える関数
def convert_handedness_for_flipped_image(handedness):
    if handedness == "Right":
        return "Left"
    elif handedness == "Left":
        return "Right"
    return handedness

#学習データをCSVに保存する関数
def save_sample(csv_path, labels, handedness, features):
    #最初に列名を記載するため，CSVファイルが存在するかを確認する．
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f: # "a"は追記
        writer = csv.writer(f)

        #CSVファイルが存在しない場合は，最初に列名を書き込む
        if not file_exists:
            header = ["hand", "thumb_label", "index_finger_label", "middle_finger_label", "ring_finger_label", "pinky_label"]
            for i in range(21):
                header += [f"x{i}", f"y{i}", f"z{i}"]
            writer.writerow(header)

        #画面反転後の表示上の左右に合わせる(自分から見て右手ならcsvにrightと記載する)
        display_hand = convert_handedness_for_flipped_image(handedness)
        sample = [display_hand] + labels[:5] + features
        writer.writerow(sample)
    return sample

#手の骨格線とランドマーク点を描画する関数
def draw_hand(frame, hand_landmarks, connections, width, height):
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

        #landmark.x，landmark.yは画像の幅・高さに対する比率(0.0-1.0)で表されるので，実際の解像度にする
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        #指先だけ赤く，他は黄色にする
        if i in FINGER_TIP_INDICES:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        cv2.circle(
            frame, #描画対象画像
            (x,y), #描画位置(円の中心)
            4, #半径
            color, #色
            -1 #塗りつぶし
        )

#データ収集時の状態を描画する関数
def draw_status_text(frame, saved_count, detected_hand_count):
    color = (0, 255, 0)

    cv2.putText(
        frame,
        "s to save / q to quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Saved: {saved_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    #1つの手だけ映っている場合のみ保存できる
    if detected_hand_count == 1:
        message = "Ready"
        color = (0, 255, 0)
    else:
        message = "Not ready: Show one hand"
        color = (0, 0, 255)

    cv2.putText(
        frame,
        message,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

def main():
    #CSVに保存した件数
    saved_count = 0

    t = 0
    i = 0
    m = 0
    r = 0
    p = 0

    options = HandLandmarkerOptions( #手検出器の設定
        base_options=BaseOptions(model_asset_path=MODEL_PATH), #モデルファイル
        running_mode=VisionRunningMode.IMAGE, #1枚ずつ処理する
        num_hands=2, #最大の手の検出数
        min_hand_detection_confidence=0.5,#手の検出信頼度
        min_hand_presence_confidence=0.5, #手ランドマークに対しての手の検出信頼度
    )

    detector = HandLandmarker.create_from_options(options) #検出器の作成
    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS #手のランドマーク同士の接続情報

    # 映像取得
    cap = cv2.VideoCapture(DEVICE)
    if not cap.isOpened():
        print("camera open error")
        return

    while True:
        # 各フレームごと，つまり画像を取得
        ret, frame = cap.read()
        if not ret:
            print("frame read error")
            break

        frame = cv2.flip(frame, 1) #画面反転．引数1が左右反転
        height, width, _ = frame.shape #画像の高さ，幅，色．今回色情報は使わない．

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #MediaPipe用の形式(RGB画像)にするために，CV2のBGR形式からRGBに変換．
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # 手の検知を行う
        hands = detector.detect(mp_image) #手の検出をする処理
        detected_hand_count = len(hands.hand_landmarks) if hands.hand_landmarks else 0

        if hands.hand_landmarks: #手が検出された場合
            # 検出した手の描画処理
            for hand_landmarks in hands.hand_landmarks: #検出された手を1つずつ取り出す
                draw_hand(frame, hand_landmarks, connections, width, height)

        draw_status_text(frame, saved_count, detected_hand_count)


        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # それぞれの指のキーが押された場合，現在の値を反転させる
        if key == ord("1"):
            t = 1 if t == 0 else 0
        if key == ord("2"):
            i = 1 if i == 0 else 0
        if key == ord("3"):
            m = 1 if m == 0 else 0
        if key == ord("4"):
            r = 1 if r == 0 else 0
        if key == ord("5"):
            p = 1 if p == 0 else 0

        cv2.putText(
            frame,
            f"thumb: {t}",
            (int(width * 0.67), 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            2
        )
        cv2.putText(
            frame,
            f"index finger: {i}",
            (int(width * 0.67), 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            2
        )
        cv2.putText(
            frame,
            f"middle finger: {m}",
            (int(width * 0.67), 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            2
        )
        cv2.putText(
            frame,
            f"ring finger: {r}",
            (int(width * 0.67), 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            2
        )
        cv2.putText(
            frame,
            f"pinky: {p}",
            (int(width * 0.67), 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            2
        )


        cv2.imshow("camera", frame)
        if key == ord("s"):
            labels = [t,i,m,r,p]
            #学習データは片手ずつ保存するため，手が検出されており，かつ手が1つだけ映っている場合のみ保存する
            if hands.hand_landmarks and len(hands.hand_landmarks) == 1:
                hand_landmarks = hands.hand_landmarks[0]
                handedness = hands.handedness[0][0].category_name #手の左右判定．handednessには"Left"か"Right"が入る
                features = landmarks_to_features(hand_landmarks)

                if features is not None:
                    sample = save_sample(CSV_PATH, labels, handedness, features)
                    saved_count += 1
                    print(f"Saved: {sample}\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
