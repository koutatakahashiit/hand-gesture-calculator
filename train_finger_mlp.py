import pandas as pd #CSVの読み込みに用いる
import numpy as np #学習データの型変換に用いる

import torch #PyTorchによる学習に用いる
import torch.nn as nn #ニューラルネットワークの実装に用いる
from torch.utils.data import Dataset, DataLoader #学習データをpytorchで扱うために用いる

from sklearn.model_selection import train_test_split #学習データとテストデータの分割に用いる
from sklearn.metrics import accuracy_score, classification_report #学習結果の評価に用いる

CSV_PATH = "data/finger_landmarks.csv" #学習データの読み込み先
MODEL_SAVE_PATH = "models/finger_mlp_model.pth" #学習済みモデルの保存先

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
RANDOM_SEED = 42

#学習データをPyTorchで扱うためのクラス(xが入力，yが正解ラベル．DataLoaderミニバッチ単位でデータを取り出すために使用)
class FingerDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32) #入力特徴量は小数なのでfloat32にする
        self.y = torch.tensor(y, dtype=torch.long) #正解ラベルはクラス番号なのでlongにする

    #データ件数を返す
    def __len__(self):
        return len(self.x)

    #index番目のデータ(入力と正解)を返す
    def __getitem__(self, index):
        return self.x[index], self.y[index]

#63次元のランドマーク特徴量から0-5本の6クラス分類を行うモデルクラス．
class FingerMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self.net(x)

#CSVから学習に使う入力データと正解ラベルを取り出す関数
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    #入力に使う63個の列名を作成する(["x0", "y0", "z0", "x1", "y1", "z1", ...]のようなもの)
    feature_cols = []
    for i in range(21):
        feature_cols += [f"x{i}", f"y{i}", f"z{i}"]

    #hand列はCSVの確認用なので学習には用いない．
    #入力と正解ラベルの定義
    x = df[feature_cols].values.astype(np.float32) #ランドマーク63次元を入力データにする
    y = df["label"].values.astype(np.int64) #0から5のラベルを正解データにする

    return x, y

#1エポック分の学習を行う関数
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x) #モデルの予測結果
        loss = criterion(logits, batch_y) #予測結果と正解ラベルの誤差

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

            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy() #最も値が大きいクラスを予測結果にする

            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    return np.array(all_labels), np.array(all_preds)

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    x, y = load_dataset(CSV_PATH)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = FingerMLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #モデルの学習
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        #10エポックごとに損失を出力
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/{EPOCHS}, loss: {avg_loss:.4f}")

    #モデルの評価
    all_labels, all_preds = evaluate(model, test_loader, device)

    acc = accuracy_score(all_labels, all_preds)
    print()
    print("test accuracy:", acc)

    print()
    print("classification report:")
    print(classification_report(all_labels, all_preds))

    #モデルの保存
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print()
    print("saved model:", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
