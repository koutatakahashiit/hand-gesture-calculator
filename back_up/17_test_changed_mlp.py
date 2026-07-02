# 現在クイズモードの実装に取り組み中

import cv2 #カメラ，画像表示．描画処理に用いる
import mediapipe as mp #手の検出・ランドマーク取得に用いる
import numpy as np #2点間距離の計算に用いる
from collections import deque, Counter #直近数フレームのうち，頻度が最も高い値を用いるために用いる(ブレを軽減させる)

import torch
import torch.nn as nn

import random #クイズモードに使用
import time #クイズモードのクールダウン計測に使用

COOL_DOWN_TIME = 2.0   # クールタイム（秒）

DEVICE = 0 #使用するカメラ番号．通常は最初に認識されたカメラが0になる

PROBLEM_NUM = 10 # クイズの問題数

#各mlpのモデルパス
T_ML_MODEL_PATH = "models/t_finger_mlp_model.pth"
I_ML_MODEL_PATH = "models/i_finger_mlp_model.pth"
M_ML_MODEL_PATH = "models/m_finger_mlp_model.pth"
R_ML_MODEL_PATH = "models/r_finger_mlp_model.pth"
P_ML_MODEL_PATH = "models/p_finger_mlp_model.pth"

MODEL_PATH = "models/hand_landmarker.task" #手検出モデルファイルのパス

HISTORY_SIZE = 10 #直近HISTORY_SIZEフレームまで指の数を保存．古いフレームは捨てる．
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

GREEN = (0, 255, 0)
RED = (0, 0, 255)

#簡易化
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

#27次元のランドマーク特徴量からの0(指が曲がっている)か1(指が伸びている)バイナリ分類を行うモデルクラス．
class FingerMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(27, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_finger_model(model_path, device):
    model = FingerMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def landmarks_to_features(hand_landmarks):
    wrist = hand_landmarks[0]

    scale = calc_dist2d(hand_landmarks[0], hand_landmarks[9])
    if scale == 0:
        return None

    features = []

    for landmark in hand_landmarks:
        x = (landmark.x - wrist.x) / scale
        y = (landmark.y - wrist.y) / scale
        z = (landmark.z - wrist.z) / scale

        features.append([x, y, z])

    return features

def extend_finger_features(features, index):
    finger_features = []

    for i in index:
        finger_features.extend(features[i])

    return finger_features

def predict_fingers_by_model(hand_landmarks, thumb_model, index_model, middle_model, ring_model, pinky_model, device):
    features = landmarks_to_features(hand_landmarks)

    if features is None:
        return None

    thumb_feature_index = [0, 1, 2, 3, 4, 5, 9, 13, 17]
    index_finger_feature_index = [0, 1, 5, 6, 7, 8, 9, 13, 17]
    middle_finger_feature_index = [0, 1, 5, 9, 10, 11, 12, 13, 17]
    ring_finger_feature_index = [0, 1, 5, 9, 13, 14, 15, 16, 17]
    pinky_feature_index = [0, 1, 5, 9, 13, 17, 18, 19, 20]

    thumb_features = extend_finger_features(features, thumb_feature_index)
    index_features = extend_finger_features(features, index_finger_feature_index)
    middle_features = extend_finger_features(features, middle_finger_feature_index)
    ring_features = extend_finger_features(features, ring_finger_feature_index)
    pinky_features = extend_finger_features(features, pinky_feature_index)

    with torch.no_grad():
        thumb_x = torch.tensor([thumb_features], dtype=torch.float32).to(device)
        index_x = torch.tensor([index_features], dtype=torch.float32).to(device)
        middle_x = torch.tensor([middle_features], dtype=torch.float32).to(device)
        ring_x = torch.tensor([ring_features], dtype=torch.float32).to(device)
        pinky_x = torch.tensor([pinky_features], dtype=torch.float32).to(device)

        thumb_prob = torch.sigmoid(thumb_model(thumb_x)).item()
        index_prob = torch.sigmoid(index_model(index_x)).item()
        middle_prob = torch.sigmoid(middle_model(middle_x)).item()
        ring_prob = torch.sigmoid(ring_model(ring_x)).item()
        pinky_prob = torch.sigmoid(pinky_model(pinky_x)).item()

        thumb_pred = int(thumb_prob >= 0.5)
        index_pred = int(index_prob >= 0.5)
        middle_pred = int(middle_prob >= 0.5)
        ring_pred = int(ring_prob >= 0.5)
        pinky_pred = int(pinky_prob >= 0.5)

    return [thumb_pred, index_pred, middle_pred, ring_pred, pinky_pred]

# 演算子にしたがって計算結果を求める関数
def calc_result(operator, left_value, right_value):
    if operator == "+":
        return left_value + right_value
    elif operator == "-":
        return left_value - right_value
    elif operator == "*":
        return left_value * right_value
    elif operator == "/":
        #0除算(分母側の右手が0)のときに未定義にする．
        if right_value == 0:
            return None
        return left_value / right_value

#二点間の距離を計算する関数(landmarkのx,yを用いる)
def calc_dist2d(p1,p2):
    p1_xy = np.array([p1.x, p1.y])
    p2_xy = np.array([p2.x, p2.y])
    return float(np.linalg.norm(p1_xy - p2_xy)) #ユークリッド距離(直線距離)


#旧方式：指の本数を数える関数
# def count_fingers(hand_landmarks):
#     wrist = hand_landmarks[0] #手首
#     extend_ratio = 1.15 #パラメータ

#     cnt = 0

#     for tip_index, base_index in FINGER_PAIRS:
#         tip = hand_landmarks[tip_index]
#         base = hand_landmarks[base_index]

#         # 手首から指先までの距離が、手首から関節までの距離より十分大きい場合、その指を立てたと判定する
#         if calc_dist2d(wrist,tip) > calc_dist2d(wrist, base) * extend_ratio:
#             cnt += 1
#     return cnt

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
            color = RED
        else:
            color = (0, 255, 255)

        cv2.circle(
            frame, #描画対象画像
            (x,y), #描画位置(円の中心)
            4, #半径
            color, #色
            -1 #塗りつぶし
        )

def draw_status_text(frame, stable_left, stable_right, operator, result):

    color = GREEN
    #左手の指の本数を描画
    left_text = stable_left if stable_left is not None else "-"
    cv2.putText(
        frame,
        f"Left : {left_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
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
        color,
        2
    )

    #演算子を描画
    cv2.putText(
        frame,
        f"operator: {operator}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )


    #合計の描画
    if stable_left is not None and stable_right is not None:
        if result is None:
            result_message = "Divide by Zero"
            color = RED
        else:
            if operator == "/":
                result_message = f"{stable_left} {operator} {stable_right} = {result:.2f}"
            else:
                result_message = f"{stable_left} {operator} {stable_right} = {result}"
    else:
        result_message = "Show both hands"

    cv2.putText(
        frame,
        result_message,
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )

#クイズモードの作成関数(ただし0-10までの値で答えられる範囲にする．)
def generate_quiz():
    while(True):
        #式に必要な数字をランダムで生成(大きい数になると0から10以下の範囲に収まりにくいので20以下で設定)
        nums = [str(random.randint(0,20)) for _ in range(random.randint(2,5))]

        operator = ["+", "*" , "-" , "/"]

        #演算子の数 = 数字の数 - 1
        operator_nums = len(nums) - 1
        #演算子をランダム選択し，生成
        selected_operator = [random.choice(operator) for _ in range(operator_nums)]

        formula = ""
        j = 0
        #式を順に数字→演算子→数字のように文字列として結合していく．
        for num in nums:
            formula += num
            if j < operator_nums:
                formula += selected_operator[j]
                j += 1

        #文字列をプログラムとして実行させる．0除算になった場合はもう一度1から式生成をやり直す．
        try:
            answer = eval(formula)
        except ZeroDivisionError:
            continue

        #式の答えが指の本数で表現可能かを判定．可能ならループから抜ける．答えが小数にならないかもここで判定．
        if (0 <= answer <= 10) and answer == int(answer):
            answer = int(answer)
            break
    return answer, formula

#クイズの問題数と式を表示．
def start_quiz(frame, width, quiz_cnt, formula):
    cv2.putText(
        frame,
        f"question: {quiz_cnt}",
        (int(width * 0.67), 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0,0,0),
        2
    )

    cv2.putText(
        frame,
        f"{formula} = ?",
        (int(width * 0.50), 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0,0,0),
        2
    )

def main():
    left_history = deque(maxlen=HISTORY_SIZE)
    right_history = deque(maxlen=HISTORY_SIZE)

    # 手がMISSING_FRAME_LIMITフレーム連続で検出されなかった場合、履歴をリセットするために用いる
    left_missing_cnt = 0
    right_missing_cnt = 0

    #演算子の選択．初期は+とする
    operator = "+"

    #クイズモードにするか
    quiz_mode = False

    options = HandLandmarkerOptions( #手検出器の設定
        base_options=BaseOptions(model_asset_path=MODEL_PATH), #モデルファイル
        running_mode=VisionRunningMode.IMAGE, #1枚ずつ処理する
        num_hands=2, #最大の手の検出数
        min_hand_detection_confidence=0.5,#手の検出信頼度
        min_hand_presence_confidence=0.5, #手ランドマークに対しての手の検出信頼度
    )

    detector = HandLandmarker.create_from_options(options) #検出器の作成
    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS #手のランドマーク同士の接続情報

    device = torch.device("cpu")

    thumb_model = load_finger_model(T_ML_MODEL_PATH, device)
    index_finger_model = load_finger_model(I_ML_MODEL_PATH, device)
    middle_finger_model = load_finger_model(M_ML_MODEL_PATH, device)
    ring_finger_model = load_finger_model(R_ML_MODEL_PATH, device)
    pinky_model = load_finger_model(P_ML_MODEL_PATH, device)

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
        hands = detector.detect(mp_image) #手の検出をする処理

        if hands.hand_landmarks: #手が検出された場合

            # 検出した両手の描画処理
            for index, hand_landmarks in enumerate(hands.hand_landmarks): #検出された手を1つずつ取り出す(今回の場合，一人なので最大2つ)

                #片手ごとに指の本数を数える．
                handedness = hands.handedness[index][0].category_name #手の左右判定．handednessには"Left"か"Right"が入る

                #距離計算
                # finger_cnt = count_fingers(hand_landmarks)

                preds = predict_fingers_by_model(
                    hand_landmarks, thumb_model,
                    index_finger_model,
                    middle_finger_model,
                    ring_finger_model,
                    pinky_model,
                    device
                    )

                if preds is None:
                    continue
                else:
                    finger_cnt = sum(preds)

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

        result = None
        if stable_left is not None and stable_right is not None:
            result = calc_result(operator, stable_left, stable_right)
        draw_status_text(frame, stable_left, stable_right, operator, result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and not quiz_mode: #クイズモード(cキー)をオンにする．初期化をここで行う．
            quiz_mode = True

            #クールダウン
            is_waiting_next_quiz = False
            last_result_time = 0.0
            result_hold_start_time = 0.0

            #問題番号
            quiz_cnt = 1
            ac_cnt = 0
            is_correct = None

            #前フレームの回答結果．回答待機時間に使用
            pre_result = None

            #式の生成．答えと式を返す
            answer, formula = generate_quiz()

        elif key == ord("+"):
            operator = "+"
        elif key == ord("-"):
            operator = "-"
        elif key == ord("*"):
            operator = "*"
        elif key == ord("/"):
            operator = "/"

        if quiz_mode:

            # 指の本数が答えに対応．よってクイズの間は演算子を+に固定．
            operator = "+"

            now = time.time()

            #現在クールダウン中か
            if is_waiting_next_quiz:
                if is_correct:
                    cv2.putText(
                        frame,
                        "Correct!",
                        (int(width * 0.50), 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        GREEN,
                        2
                    )
                else:
                    cv2.putText(
                        frame,
                        "Wrong answer!",
                        (int(width * 0.50), 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        RED,
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Answer is {answer}",
                        (int(width * 0.50), 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        GREEN,
                        2
                    )

                #問題終了後の結果発表
                if quiz_cnt >= PROBLEM_NUM:
                    cv2.putText(
                        frame,
                        f"{ac_cnt} / {PROBLEM_NUM} Correct",
                        (int(width * 0.50), 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0,0,0),
                        2
                    )


                #結果が出てからの経過時間 >= クールダウンなら次の問題に進む．
                if now - last_result_time >= COOL_DOWN_TIME:
                    is_waiting_next_quiz = False

                    # PROBLEM_NUM個答えたならクイズモード終了
                    if quiz_cnt >= PROBLEM_NUM:
                        quiz_mode = False
                    else:
                        result_hold_start_time = now
                        quiz_cnt += 1
                        answer, formula = generate_quiz()

            #問題を表示している状態．
            else:
                start_quiz(frame, width, quiz_cnt, formula)
                if result is not None:
                    # 回答が変わった瞬間だけ，計測開始時刻を更新する
                    if pre_result != result:
                        pre_result = result
                        result_hold_start_time = now
                    else:
                        cv2.putText(
                            frame,
                            f"Hold {now - result_hold_start_time:.1f}s / {COOL_DOWN_TIME:.1f}s",
                            (int(width * 0.50), 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0,0,0),
                            2
                        )

                    # 同じ回答が2秒続いたら正誤判定する
                    if now - result_hold_start_time >= COOL_DOWN_TIME:
                        #正解
                        if result == answer:
                            is_correct = True
                            ac_cnt += 1
                        #不正解
                        else:
                            is_correct = False

                        #回答が出たので，次の問題の表示までクールダウンを発生させる．
                        last_result_time = now
                        is_waiting_next_quiz = True
                else:
                    result_hold_start_time = now

        cv2.imshow("Camera", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()