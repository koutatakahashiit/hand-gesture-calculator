import cv2 #カメラ，画像表示．描画処理に用いる
import mediapipe as mp #手の検出・ランドマーク取得に用いる
import numpy as np #2点間距離の計算に用いる
from collections import deque, Counter #直近数フレームのうち，頻度が最も高い値を用いるために用いる

DEVICE = 0 #使用するカメラ番号．通常は最初に認識されたカメラが0になる
MODEL_PATH = "models/hand_landmarker.task" #手検出モデルファイルのパス
HISTORY_SIZE = 5 #直近HISTORY_SIZEフレームまで指の数を保存．古いフレームは捨てる．
MISSING_FRAME_LIMIT = 3
FINGER_TIP_INDICES = [4, 8, 12, 16, 20] #指先のランドマーク番号

#各指先の関節とのランドマーク番号．距離計算に用いる．
FINGER_PAIRS = [
    (4,2), #親指．4が指先，2が関節
    (8,6), #人差し指．8が指先，6が関節
    (12,10), #中指．12が指先，10が関節
    (16,14), #薬指．16が指先，14が関節
    (20,18) #小指．20が指先，18が関節
]

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

#指の本数を数える関数(改良の余地あり．距離計算のみのため)
# 試してみたいもの：ランドマークを入力とする深層学習
def count_fingers(hand_landmarks):
    wrist = hand_landmarks[0] #手首
    extend_ratio = 1.15 #パラメータ
    
    cnt = 0
    
    for tip_index, base_index in FINGER_PAIRS:
        tip = hand_landmarks[tip_index]
        base = hand_landmarks[base_index]
        
        # 手首から指先までの距離が、手首から関節までの距離より十分大きい場合、その指を立てたと判定する
        if calc_dist2d(wrist,tip) > calc_dist2d(wrist, base) * extend_ratio:
            cnt += 1
    return cnt

#履歴のうち，最も頻度の高い値を返す関数．
def get_common_value(history, default=None):
    if len(history) == 0: #履歴が空の場合
        return default
    
    #履歴が存在するので，一番頻度の高い値を返す
    return Counter(history).most_common(1)[0][0]

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

        #landmark.x，landmark.yは画像の幅・高さに対する比率(0.0-1.0)で表される
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

def draw_status_text(frame, stable_left, stable_right):
    
    #左手の指の本数を描画
    left_text = stable_left if stable_left is not None else "-"
    cv2.putText(
        frame,
        f"Left : {left_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    #右手の指の本数を描画
    right_text = stable_right if stable_right is not None else "-"
    cv2.putText(
        frame,
        f"Right: {right_text}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    
    #合計の描画
    if stable_left is not None and stable_right is not None:
        total = stable_left + stable_right
        total_message = f"{stable_left} + {stable_right} = {total}"
    else:
        total_message = "Show both hands"
    cv2.putText(
        frame,
        total_message,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

def main():
    left_history = deque(maxlen=HISTORY_SIZE)
    right_history = deque(maxlen=HISTORY_SIZE)

    # 手がMISSING_FRAME_LIMITフレーム連続で検出されなかった場合、履歴をリセットするために用いる
    left_missing_cnt = 0
    right_missing_cnt = 0

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
        
        #片手ごとの指の本数．左右の手が検出されないケースもあるため，初期値はNoneにする．
        left_cnt = None
        right_cnt = None
        
        #右手，左手が検出されたかどうか
        right_seen = False
        left_seen = False
        
        frame = cv2.flip(frame, 1) #画面反転．引数1が左右反転
        height, width, _ = frame.shape #画像の高さ，幅，色．今回色情報は使わない．

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #MediaPipe用の形式(RGB画像)にするために，CV2のBGR形式からRGBに変換．
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # 手の検知を行う
        result = detector.detect(mp_image) #手の検出をする処理

        if result.hand_landmarks: #手が検出された場合

            # 検出した両手の描画処理
            for index, hand_landmarks in enumerate(result.hand_landmarks): #検出された手を1つずつ取り出す(今回の場合，一人なので最大2つ)
                
                #片手ごとに指の本数を数える．
                handedness = result.handedness[index][0].category_name #手の左右判定．handednessには"Left"か"Right"が入る
                finger_cnt = count_fingers(hand_landmarks)
                if handedness == "Right": # 左右反転した画像をMediaPipeに渡しているため、表示上の左右に合わせて入れ替える
                    left_cnt = finger_cnt
                    left_history.append(left_cnt)
                    left_seen = True
                elif handedness == "Left":
                    right_cnt = finger_cnt
                    right_history.append(right_cnt)
                    right_seen = True

                #手の線と点の描画
                draw_hand(frame, hand_landmarks, connections, width, height)

        # 各フレームにおいて右手，左手が検出できたか．
        if left_seen:
            left_missing_cnt = 0
        else:
            left_missing_cnt += 1

        if right_seen:
            right_missing_cnt = 0
        else:
            right_missing_cnt += 1
        
        #右手 or 左手をMISSING_FRAMEフレーム検出しなかった場合，履歴を削除する．
        if left_missing_cnt >= MISSING_FRAME_LIMIT:
            left_history.clear()
            left_missing_cnt = 0
        if right_missing_cnt >= MISSING_FRAME_LIMIT:
            right_history.clear()
            right_missing_cnt = 0
        
        #両手の合計を描画
        stable_left = get_common_value(left_history)
        stable_right = get_common_value(right_history)
        
        draw_status_text(frame, stable_left, stable_right)

        cv2.imshow("Camera", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()