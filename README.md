# CNN MINST
CNN을 이용한 MNIST 계산 : 손으로 쓴 숫자 이미지를 인식하는데 CNN를 활용하고자 한다.

사람이 손으로 쓰거나 기계로 인쇄한 문서 이미지를 스캐닝하면 이미지가 문자로 표현된 문서파일로 바뀌는 기술이 광학문자인식(OCR) 기술이다. OCR은 인공지능이나 시각기계의 초기형태라 본다. 기계가 선의 패턴을 보고 같은 꼴을 한 글자로 변환시켜주는 기술이다. 이때 기계는 활자가 가진 의미를 알지 못한다. 다만, 지정된 언어의 활자 중에서 모양이 가장 일치하는 글자로 변환해 주는 역할만을 한다. 문장 중에서 인접한 단어가 오는 것이 어색한지를 분석하는 기술은 맞춤법을 기초로 한 문장해석기술이다. 예전엔 OCR 소프트웨어를 스캐너를 구매하면 함께 제공받았는데 최근엔 인터넷에서 공짜로 OCR 서비스를 받을 수 있다. 

OCR은 간단한 기술로 인식되지만 신경망(Neural Network) 지능기술이 적용된다. 새로운 알고리즘의 인공지능을 평가하는 벤치마크 대상으로 삼는 표준 필기체 숫자가 있다. 미국표준과학기술연구소는 인공지능의 학습기법과 인식기술을 개발하는 사람들이 활용할 수 있도록 필기체 숫자 데이터베이스(MNIST)를 제공하고 있다. MNIST는 훈련용과 시험용이 있다. 훈련용으로 제공되는 약 6만개의 패턴 세트는 250여명의 작가, 고등학생, 인구조사국 직원 등 500여명 이상의 필기체가 포함되어 있어서 다양한 글자꼴을 인식하는 효과가 높다. 타 제품과 비교하기 위한 시험용은 1만개의 세트가 있다. 필기체 숫자인식의 오류 발생률이 최신기술에 의하면 0.23% 이하라고 한다.


## 숫자 문자인식에 도전 (MNIST)
여기에서는 손으로 쓴 글자 이미지를 인식하는 데 CNN을 활용하고자 한다. 문자 인식에서 가장 쉬운 것이 숫자 인식이다. 대량의 이미지 데이터를 학습시켜 0-9 사이의 숫자 하나로  분류하는 것을 말한다.

또한 다행이도 대량의 숫자 이미지 데이터를 공개하고 있는 웹사이트가 있다. 바로 MNIST라고 하는 사이트다. 이 사이트에서는 0-9까지의 숫자 이미지 데이터 7만 개를 공개하고 있어 기계 학습과 패턴 인식에 사용할 수 있다. 각  이미지 데이터는 가로 세로 28 X 28픽셀의 크기다.

```python
The MNIST database of handwritten digits  
http://yann.lecun.com/exdb/mnist/
```
## CNN에서 학습 모델을 만드는 시나리오
방대한 7만 개의 수기 숫자 이미지 데이터를 학습시켜서 높은 정확도의 문자 인식에 도전해 보자. 작업을  시작하기 전에 먼저 CNN으로 문자 인식시킬 때의 절차를 알아보도록 하자. CNN을 사용하는 기본적인 절차는 다음과 같다.

```python  
  1. 패턴을 학습시킨다. 
  2. 학습 데이터에 기초하여 수기 문자를 판정한다.
```

이중에서 수기 이미지들을 학습시키는 절차는 다음과 같다.

```python
  1. MNIST의 수기 숫자 데이터를 다운로드
  2. 데이터를 28 X 28 픽셀로 변환
  3. 픽셀자료로 변환한 자료를 바탕으로 학습데이터를 작성
  4. 학습 데이터를 CNN에 학습시키고 모델을 작성
```

## CNN으로 MNIST 테스트해보기

그럼 CNN를 사용해서 MNIST 손글씨  숫자를 분류해보겠습니다. 다음은 CNN Keras로 MNIST 손글씨 숫자를 분류하는 프로그램입니다. 


```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


# MNIST 데이터 읽어 들이기
(X_train, y_train), (X_test, y_test) = mnist.load_data()



# 데이터를 28 X 28 픽셀 float32 자료형으로 변환하고 정규화하기
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255



# 레이블 데이터를 0-9까지의 카테고리를 나타내는 배열로 변환하기
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

"""
0 → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
1 → [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
2 → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
etc...
"""

print(X_train.shape); print(X_test.shape)



# CNN 모델 구조 정의하기
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape = (28, 28, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))

# 모델 구축하기
rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

# 데이터 훈련하고 저장하기, 만약 저장 되어 있으면 그 자료를 사용하기
hdf5_file="./mnist-cnn-model.hdf5"

if os.path.exists(hdf5_file):
    model.load_weights(hdf5_file)
else:
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=100, epochs=3)
    model.save_weights(hdf5_file)

# 테스트 데이터로 평가하기
score = model.evaluate(X_test, Y_test, batch_size=32)
print(score[0], score[1])

# 예측이 틀린 결과 또는 맞힌 결과를 찾아보기
y_pred = model.predict_classes(X_test)
false_preds = [im for im in zip(X_test,y_pred,y_test) if im[1] != im[2]]
true_preds = [im for im in zip(X_test,y_pred,y_test) if im[1] == im[2]]


# 예측이 틀린 결과를 plot하기
plt.figure(figsize=(10, 10))
for ind, val in enumerate(false_preds[:30]):
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.subplot(10, 10, ind + 1)
    im = 1 - val[0].reshape((28,28))
    plt.axis("off")
    plt.text(0, 0, val[2], fontsize=14, color='blue')
    plt.text(8, 0, val[1], fontsize=14, color='red')
    plt.imshow(im, cmap='gray')
```



```python
# backend로 TensorFlow(tf) or Theano(th)  
from keras import backend as K


# 훈련된 Filter를 plot하기
W = model.layers[0].get_weights()[0]

print(W.shape)
if K.image_dim_ordering() == 'tf':
    # (nb_filter, nb_channel, nb_row, nb_col)
    W = W.transpose(3, 2, 0, 1)
    nb_filter, nb_channel, nb_row, nb_col = W.shape
    print(W.shape)

plt.figure(figsize=(10, 10), frameon=False)
for i in range(32):
        im = W[i][0]
        plt.subplot(6, 6, i + 1)
        plt.axis('off')
        plt.imshow(im, cmap='gray')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
```




```python

model.summary(): CNN 모델 구조

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 32)        320       
_________________________________________________________________
activation_1 (Activation)    (None, 28, 28, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                62730     
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0         
=================================================================
Total params: 63050
Trainable params: 63050
Non-trainable params: 0
_________________________________________________________________
None

Result : loss = 0.0534754661273, accuracy = 0.983300005198(98%) 
```



## 결론

Keras가 backend로 TensorFlow 또는 Theano를 사용하고 있어서 자료를 저장함에 헷갈리는 부분이 있는 것은 분명하다.
하나로 통일 하거나 backend로 선택하면 자동으로 바뀌는 것이 user들의 혼란을 줄일수 있는 방법이 될것 같다.




