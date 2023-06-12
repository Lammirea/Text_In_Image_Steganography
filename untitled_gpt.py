# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:30:27 2023

@author: Venya
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Сначала надо преверить видит ли библиотека видеокарту
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


#генерация случайного изобржаения
def rand_img(size):
    return np.random.randint(0, 256, size) / 255.0

#генерация текста заданной длины
def rand_sentence(len, max):
    return np.random.randint(0, max, len)

#One-hot encodes a text into a list of word indexes of size n.
def onehot(sentece, max):
    onehot = np.zeros((len(sentece), max))
    for i, v in enumerate(sentece):
        onehot[i, v] = 1
    return onehot

#функция для генерации изображений заданных размером и текстов
#нужно для обучения нейронки
def data_generator(image_size, sentence_len, sentence_max, batch_size=32):
    while True:
        x_img = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
        x_sen = np.zeros((batch_size, sentence_len))
        y_img = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
        y_sen = np.zeros((batch_size, sentence_len, sentence_max))
        for i in range(batch_size):
            img = rand_img(image_size)
            sentence = rand_sentence(sentence_len, sentence_max)
            sentence_onehot = onehot(sentence, sentence_max)
            x_img[i] = img
            x_sen[i] = sentence
            y_img[i] = img
            y_sen[i] = sentence_onehot
        yield [x_img, x_sen], [y_img, y_sen]
        
#задаём размер изображения
image_shape=(400,400,3)

#а также макисмальную длину символов в тексте
sentence_len=400
max_word=1024

#энкодер для предложения
input_sen = Input((sentence_len,))

#надо подавать матрицу в embedding
embed_sen = Embedding(max_word, 400, input_length = 400)(input_sen) #TensorShape([None, 100, 100])
flat_emb_sen = Flatten()(embed_sen) #TensorShape([None, 10000])
flat_emb_sen = Reshape((image_shape[0], image_shape[1], 1))(flat_emb_sen) #TensorShape([None, 100, 100, 1])

#свёрточный автоэнкодер для изображения
input_img = Input(image_shape)
trans_input_img = Conv2D(128, 1, activation='relu', padding = 'same')(input_img)
x1 = Conv2D(64,1, activation='relu', padding = 'same')(trans_input_img)
x2 = Conv2D(32,1, activation='relu', padding = 'same')(x1)

#объединяем свёрточный энкодер изображения и энкодер текста
enc_input = Concatenate(axis=-1)([flat_emb_sen, x2])


out_img = Conv2D(3, 1, activation='relu', padding = 'same',name='image_reconstruction')(enc_input)#скрытый слой для восстановления изображения

#декродер для извлечения предложения
decoder_model = Sequential(name="sentence_reconstruction")
decoder_model.add(Conv2D(1, 1, input_shape=(400, 400, 3)))
decoder_model.add(Reshape((sentence_len, 400)))
decoder_model.add(TimeDistributed(Dense(max_word, activation="softmax")))
out_sen = decoder_model(out_img)

#задаём размер изображения (ширина,высота и количество цветовых каналов)
image_shape = (400,400,3)
sentence_len = 400 


#создаём автоэнкодер
model = Model(inputs=[input_img, input_sen], outputs=[out_img, out_sen])
model.compile(optimizer=Adam(0.002), loss=['mae', 'categorical_crossentropy'], metrics={'sentence_reconstruction': 'categorical_accuracy'})
print(model.summary())
#передаём размеры изображения,длину предложения и количество пакетов для одной эпохи
#всё это необхожимо для обучения нашей НС
gen = data_generator(image_shape, sentence_len, max_word,6)
#обучаем модель
#самый лучший вариант из всех используемых. Он затёрся,завтра надо возобновить
model.fit(gen, epochs=520, steps_per_epoch=348, callbacks=[ModelCheckpoint("mae_cat_3conv_520ep_348steps_6batch_model_400im.h5", monitor="loss", save_best_only=True),TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False)])
#надо сделать модель с 6 батчами на 520 эпох. Завтра


#функция для кодирования нашего сообщения сначала в двоичный формат,а потом в ascii
# def encode_msg(message, sentence_len):
    
#     if sentence_len <= 400:
#         sen = np.zeros([1, 400]) #создаём массив из 400 элементов,заполненный нулями
#         for i, a in enumerate(message.encode("ascii")):
#             sen[0, i] = a

#     else:
#         if sentence_len % 400 == 0:
#            sen = np.zeros([sentence_len/400,400])
#            enc = np.zeros((1,400*(len(messag)//400)))
           
#            for i, a in enumerate(messag.encode("ascii")):
#                enc[0, i] = a
               
#            count = 0
#            for i in range(sentence_len/400):
#                for j in range(400):
#                    sen[i,j]= enc[0,count]
#                    count+=1
#            # msg = np.array_split(sen, sentence_len/100)
#         else:
#            sen = np.zeros([(sentence_len//400)+1,400])
#            enc = np.zeros((1,400*(len(messag)//400 + 1)))

#            for i, a in enumerate(messag.encode("ascii")):
#                enc[0, i] = a
               
#            count = 0
#            for i in range(sentence_len//400 + 1):
#                for j in range(400):
#                    sen[i,j]= enc[0,count]
#                    count+=1
#            # msg = np.array_split(sen, sentence_len//100 + 1)        
#     return sen
    
def encode_msg(message, sentence_len):
   sen = np.zeros([1, 500]) #создаём массив из 400 элементов,заполненный нулями
   for i, a in enumerate(message.encode("ascii")):
       sen[0, i] = a
      
   return sen

#декодируем сообщение,которое извлекли из изображения
def ascii_decode(message):
    return ''.join(chr(int(a)) for a in message[0].argmax(-1))


#из этого файла мы получим текст,который запишем в изобржаение
with open('./data/secret.txt', 'r') as f:
    str1 = f.read()

#загружаем обученную модель
model = load_model('mae_cat_3conv_550ep_16steps_16batch_500img.h5')
encoder = Model(model.input, model.get_layer('image_reconstruction').output)

#для декодрирования нам необходим слой,который извлекает предложение из изображения
decoder = model.get_layer('sentence_reconstruction')

image_to_encode = "./data/train/2233.jpg"
#загружаем изображение в формате массива float32
img = np.expand_dims(img_to_array(load_img(image_to_encode, target_size=(500, 500))) / 255.0, axis=0)

#сначала кодируем сообщение из файла
sen = encode_msg(str1,len(str1))

y = encoder.predict([img, sen])
y_hat = decoder.predict(y)

#создаём лист,который содержит исходное и закодированное изображение для визуального сравнения
img_to_show=[load_img(image_to_encode), y[0]]
titles=["Input image", "Image after encoding"]

#отобразим полученное изображение отдельно от исходного
from PIL import Image
formatted = (img_to_show[1] * 255 / np.max(img_to_show[1])).astype('uint8')
img = Image.fromarray(formatted)
img.show()
img.resize((500,500))
img.save("encode500.jpg")


#создаём фигуру,которая содержит два изображения (исходное и закодированное) 
plt.figure(num=1, figsize=(4, 2))
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(img_to_show[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(titles[i]) #создаём подписи над каждым изображением
    plt.savefig("outputMAE_categ_2233_ep_348steps_16batch_500img.jpg")
plt.show()

#выводим декодированное сообщение
print("\n\n\nDecode message: "+ascii_decode(y_hat))

#записываем его в новый файл
with open('./data/decode.txt', 'w') as f:
    f.write(ascii_decode(y_hat))
    