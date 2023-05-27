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
import torch
from keras.optimizers import Adam
from lpips import LPIPS

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(f"# Using device: {device}")

torch.cuda.empty_cache()

def rand_img(size):
    return np.random.randint(0, 256, size) / 255.0

def rand_sentence(len, max):
    return np.random.randint(0, max, len)


def onehot(sentece, max):
    onehot = np.zeros((len(sentece), max))
    for i, v in enumerate(sentece):
        onehot[i, v] = 1
    return onehot

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
        

image_shape=(100, 100, 3)

sentence_len=100
max_word=256

input_img = Input(image_shape)
input_sen = Input((sentence_len,))
embed_sen = Embedding(max_word, 100)(input_sen)
flat_emb_sen = Flatten()(embed_sen)
flat_emb_sen = Reshape((image_shape[0], image_shape[1], 1))(flat_emb_sen)

trans_input_img = Conv2D(64, 1, activation='relu', padding = 'same')(input_img)
x1 = Conv2D(32,1, activation='relu', padding = 'same')(trans_input_img)
x2 = Conv2D(16,1, activation='relu', padding = 'same')(x1)

enc_input = Concatenate(axis=-1)([flat_emb_sen, trans_input_img,x1,x2])

out_img = Conv2D(3, 1, activation='relu', padding = 'same',name='image_reconstruction')(enc_input)

decoder_model = Sequential(name="sentence_reconstruction")
decoder_model.add(Conv2D(1, 1, input_shape=(100, 100, 3)))
decoder_model.add(Reshape((sentence_len, 100)))
decoder_model.add(TimeDistributed(Dense(max_word, activation="softmax")))
out_sen = decoder_model(out_img)

model = Model(inputs=[input_img, input_sen], outputs=[out_img, out_sen])
model.compile(optimizer=Adam(0.001), loss=['mae', 'categorical_crossentropy'], metrics={'sentence_reconstruction': 'categorical_accuracy'})
encoder_model = Model(inputs=[input_img, input_sen], outputs=[out_img])

image_shape = (100, 100, 3)
sentence_len = 100
max_word = 256

symbols = 0

with open('./data/secret.txt', 'r') as f:
    str1 = f.read()
    
symbols = len(str1)
gen = data_generator(image_shape, sentence_len, max_word, 64)

model.fit(gen, epochs=100, steps_per_epoch=348, callbacks=[ModelCheckpoint("best_model_mae_cat_3conv_100ep.h5", monitor="loss", save_best_only=True),TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False)])


def encode_msg(message, sentence_len):
    sen = np.zeros((1, sentence_len)) #создаём массив из 100 элементов,заполненный нулями
    for i, a in enumerate(message.encode("ascii")):
        sen[0, i] = a
    
    return sen


def ascii_decode(message):
    return ''.join(chr(int(a)) for a in message[0].argmax(-1))


model = load_model('best_model_mae_cat_3conv_100ep.h5')
encoder = Model(model.input, model.get_layer('image_reconstruction').output)
decoder = model.get_layer('sentence_reconstruction')

img = np.expand_dims(img_to_array(load_img("./data/train/2232.jpg", target_size=(100, 100))) / 255.0, axis=0)

sen = encode_msg(str1,100)
y = encoder.predict([img, sen])
y_hat = decoder.predict(y)

img_to_show=[load_img("./data/train/2232.jpg", target_size=(100,100)), y[0]]
titles=["Input image", "Image after encoding"]

plt.figure(num=1, figsize=(4, 2))
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(img_to_show[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(titles[i])
    plt.savefig("outputMAE_categ_2232_100ep.jpg")
plt.show()

print("\n\n\nDecode message: "+ascii_decode(y_hat))

with open('./data/decode.txt', 'w') as f:
    f.write(ascii_decode(y_hat))
