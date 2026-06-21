#0から10までの範囲で答えられる計算式の作成プログラム．今後はこれを組み込んでクイズモードにしたい．
import random

PROBLEM_NUM = 10 #問題数
ac_cnt = 0

for i in range(1, PROBLEM_NUM + 1):
    while(True):
        #式に必要な数字をランダムで生成(大きい数になると0から10以下の範囲に収まりにくいので20以下で設定)
        nums = [str(random.randint(0,20)) for _ in range(random.randint(2,5))]

        operators = ["+", "*" , "-" , "/"]

        #演算子の数 = 数字の数 - 1
        operator_nums = len(nums) - 1
        #演算子をランダム選択し，生成
        selected_operator = [random.choice(operators) for _ in range(operator_nums)]

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

        #式の答えが指の本数で表現可能かを判定．可能ならループから抜ける．
        if (0 <= answer <= 10) and answer == int(answer):
            answer = int(answer)
            break

    print(f"{i}問目", end="  ")
    print(formula + "= ?")
    n = int(input())
    if n == answer:
        print("正解! 100点!\n")
        ac_cnt += 1
    else:
        print("不正解! -100点!")
        print(f"正解は{answer}\n")

print(f"{PROBLEM_NUM}問中{ac_cnt}問正解!")

if ac_cnt <= 3:
    print("もっと頑張ろう!")
elif ac_cnt <= 6:
    print("いいね!")
elif ac_cnt <= 9:
    print("あともう少し!")
else:
    print("素晴らしい!")