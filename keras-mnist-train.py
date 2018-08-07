#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.datasets import mnist
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras import optimizers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

# 学習データと評価データの読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784)[:6000]
X_test  = X_test.reshape(X_test.shape[0], 784)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test  = to_categorical(y_test)[:1000]

# モデルを管理するインスタンスを生成
model = Sequential()
# 入力ユニット数は784、一つ目の全結合層の出力ユニット数は256
model.add(Dense(256, input_dim=784))
model.add(Activation("sigmoid"))
# 二つ目の全結合層の出力ユニット数は128
model.add(Dense(128))
model.add(Activation("relu"))
# 三つ目の全結合層(出力層)の出力ユニット数は10
model.add(Dense(10))
model.add(Activation("softmax"))

# 学習処理の設定
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

# 学習を実行する
# verbose: 学習の進捗度合いを表示(1で表示)
# epochs: 繰り返し数
history = model.fit(X_train, y_train, verbose=1, epochs=3)

plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

'''
# モデル構造の出力
plot_model(model, "model125.png", show_layer_names=False)
# モデル構造の可視化
image = plt.imread("model125.png")
plt.figure(dpi=150)
plt.imshow(image)
plt.show()
'''
