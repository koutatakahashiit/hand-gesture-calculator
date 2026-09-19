import pandas as pd #CSVの読み込みに用いる
import numpy as np #学習データの型変換に用いる

import torch #PyTorchによる学習に用いる
import torch.nn as nn #ニューラルネットワークの実装に用いる
from torch.utils.data import Dataset, DataLoader #学習データをpytorchで扱うために用いる

from sklearn.model_selection import train_test_split #学習データとテストデータの分割に用いる
from sklearn.metrics import accuracy_score, classification_report #学習結果の評価に用いる

CSV_PATH = "data/hand_landmarks_and_states.csv" #学習データの読み込み先
T_MODEL_SAVE_PATH = "models/t_finger_mlp_model.pth"
I_MODEL_SAVE_PATH = "models/i_finger_mlp_model.pth"
M_MODEL_SAVE_PATH = "models/m_finger_mlp_model.pth"
R_MODEL_SAVE_PATH = "models/r_finger_mlp_model.pth"
P_MODEL_SAVE_PATH = "models/p_finger_mlp_model.pth"

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
RANDOM_SEED = 42

#学習データをPyTorchで扱うためのクラス(xが入力，yが正解ラベル．DataLoaderミニバッチ単位でデータを取り出すために使用)
class FingerDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32) #入力特徴量は小数なのでfloat32にする
        self.y = torch.tensor(y, dtype=torch.float32) #ラベルは確率になるのでfloat32にする

    #データ件数を返す
    def __len__(self):
        return len(self.x)

    #index番目のデータ(入力と正解)を返す
    def __getitem__(self, index):
        return self.x[index], self.y[index]

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

#各指用のデータセットの作成
def load_finger_dataset(csv_path, sel):
    df = pd.read_csv(csv_path)

    if sel == "thumb":
        #親指用の特徴量
        finger_feature_index = [0, 1, 2, 3, 4, 5, 9, 13, 17]

    elif sel == "index_finger":
        #人差し指の特徴量
        finger_feature_index = [0, 1, 5, 6, 7, 8, 9, 13, 17]

    elif sel == "middle_finger":
        #中指の特徴量
        finger_feature_index = [0, 1, 5, 9, 10, 11, 12, 13, 17]

    elif sel == "ring_finger":
        #薬指の特徴量
        finger_feature_index = [0, 1, 5, 9, 13, 14, 15, 16, 17]

    elif sel == "pinky":
        #小指の特徴量
        finger_feature_index = [0, 1, 5, 9, 13, 17, 18, 19, 20]

    feature_cols = []
    for i in finger_feature_index:
        feature_cols += [f"x{i}", f"y{i}", f"z{i}"]

    #hand列はCSVの確認用なので学習には用いない．
    #入力と正解ラベルの定義
    x = df[feature_cols].values.astype(np.float32) #ランドマーク27次元を入力データにする
    y = df[sel + "_label"].values.astype(np.int64) #0(曲げている)，1(伸ばしている)のラベルを正解データにする

    return x, y

#1エポック分の学習を行う関数
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x) #モデルの予測結果
        loss = criterion(logits.squeeze(1), batch_y) #予測結果と正解ラベルの誤差. (32, 1) → (32,) に変換

        optimizer.zero_grad() #前回の勾配を削除する
        loss.backward() #誤差をもとに勾配を計算する
        optimizer.step() #重みを更新する

        total_loss += loss.item() * batch_x.size(0)#バッチごとのlossをデータ数分だけ加算する

    return total_loss / len(train_loader.dataset) #1データあたりの平均lossを返す

#テストデータで評価を行う関数
def evaluate(model, test_loader, device):
    model.eval() #Dropoutを無効化して評価モードにする

    all_preds = []
    all_labels = []

    with torch.no_grad(): #重み更新を無効化する
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            logits = model(batch_x).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy() #確率が0.5以上なら1と判定

            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    return np.array(all_labels), np.array(all_preds)

#各指用のmlp作成
def train_and_evaluate_finger(finger_name, csv_path, model_save_path, device):
    line = "=" * 10
    print(f"{line} start {finger_name} mlp {line}")
    print()

    x, y = load_finger_dataset(csv_path, finger_name)

    #学習用データとテスト用データに分割する(20%がテストデータ．正解ラベルの比率を保つためにstratify=yとする)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    train_dataset = FingerDataset(x_train, y_train)
    test_dataset = FingerDataset(x_test, y_test)

    #BATCH_SIZEずつデータを取り出して学習するためにDataLoaderを作成する
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    #モデル定義
    model = FingerMLP().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #モデルの学習
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        #10エポックごとに損失を出力
        if (epoch + 1) % 10 == 0:
            print(f"{finger_name} epoch {epoch + 1}/{EPOCHS}, loss: {avg_loss:.4f}")

    #モデルの評価
    all_labels, all_preds = evaluate(model, test_loader, device)

    acc = accuracy_score(all_labels, all_preds)
    print()
    print(f"{finger_name}_test accuracy:", acc)

    print()
    print(f"{finger_name}_classification report:")
    print(classification_report(all_labels, all_preds))

    #モデルの保存
    torch.save(model.state_dict(), model_save_path)
    print()
    print(f"saved {finger_name} model:", model_save_path)
    print()

    return acc

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print()

    #親指専用のmlpを作成
    thumb_acc = train_and_evaluate_finger("thumb", CSV_PATH, T_MODEL_SAVE_PATH, device)

    #人差し指専用のmlpを作成
    index_finger_acc = train_and_evaluate_finger("index_finger", CSV_PATH, I_MODEL_SAVE_PATH, device)

    #中指専用のmlpを作成
    middle_finger_acc = train_and_evaluate_finger("middle_finger", CSV_PATH, M_MODEL_SAVE_PATH, device)

    #薬指専用のmlpを作成
    ring_finger_acc = train_and_evaluate_finger("ring_finger", CSV_PATH, R_MODEL_SAVE_PATH, device)

    #小指専用のmlpを作成
    pinky_acc = train_and_evaluate_finger("pinky", CSV_PATH, P_MODEL_SAVE_PATH, device)

    line = "=" * 10
    print(f"{line} result {line}")
    print()
    print(f"thumb_acc: {thumb_acc:.4f}")
    print(f"index_finger_acc: {index_finger_acc:.4f}")
    print(f"middle_finger_acc: {middle_finger_acc:.4f}")
    print(f"ring_finger_acc: {ring_finger_acc:.4f}")
    print(f"pinky_acc: {pinky_acc:.4f}")

if __name__ == "__main__":
    main()
