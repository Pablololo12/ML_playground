#!/usr/bin/env python3
# Copyright (c) 2019, ARM Limited and Contributors
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import datetime
from matplotlib import pyplot
from Model import deepHotDog
import numpy as np
import os
import random
import sklearn.model_selection as ms
import tensorflow as tf
import time

HEIGHT = 224
WIDTH = 224
CHANNELS = 3
BATCH_SIZE = 5 
EPOCHS = 5

CLASS_0 = "hotdog"
CLASS_1 = "office"

CHECKPOINT_PATH = os.path.join("checkpoint","cp.ckpt")
LOG_PATH = os.path.join("log","fit",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def parse_arguments():
    parser = argparse.ArgumentParser(description="deepHotDog")
    parser.add_argument('-t', help="Train Model", action='store_true')
    parser.add_argument('-e', help="Eval Model", action='store_true')
    parser.add_argument('-i', help="Show intermediate layers", action='store_true')
    parser.add_argument('-f', help="Folder with folders")
    parser.add_argument('-c', help="Convert Model to tflite", action='store_true')
    return parser.parse_args()

def _get_img(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = (tf.cast(img, tf.float32)/127.5) - 1
    img = tf.image.resize(img, (HEIGHT, WIDTH))
    return img, label

def get_data(folder):
    hot_files = [os.path.join(folder,CLASS_0,f) for f in os.listdir(os.path.join(folder,CLASS_0))]
    random.shuffle(hot_files)
    hot_labels = [0 for x in range(len(hot_files))]

    not_files = [os.path.join(folder,CLASS_1,f) for f in os.listdir(os.path.join(folder,CLASS_1))]
    random.shuffle(not_files)
    not_labels = [1 for x in range(len(not_files))]

    filenames = hot_files+not_files
    labels = hot_labels+not_labels

    train_filenames, val_filenames, train_labels, val_labels = ms.train_test_split(
                            filenames, labels, train_size=0.8)
    train_data = tf.data.Dataset.from_tensor_slices(
        (tf.constant(train_filenames), tf.constant(train_labels)))
    val_data = tf.data.Dataset.from_tensor_slices(
        (tf.constant(val_filenames), tf.constant(val_labels)))
    train_data = (train_data.map(_get_img).shuffle(buffer_size=1000).batch(BATCH_SIZE))
    val_data = (val_data.map(_get_img).shuffle(buffer_size=1000).batch(BATCH_SIZE))
    return train_data, val_data

def train_model(model, folder):
    print("Getting data")
    train_data, val_data = get_data(folder)
    print(train_data)

    print("Training model")
    #model.load_weights(CHECKPOINT_PATH)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH,
                                                monitor='val_loss',
                                                verbose=0,
                                                save_best_only=False,
                                                save_weights_only=False,
                                                mode='auto', save_freq=500)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH, histogram_freq=1, write_graph=True, write_images=True)
    model.fit(x=train_data,
            epochs=EPOCHS,
            verbose=1,
            validation_data=val_data,
            callbacks=[checkpoint, tensorboard_callback])
    tensorboard_callback.set_model(model)
    model.save(os.path.join("model", "1"))

def show_filter(filters, r, c):
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    fig = pyplot.figure(figsize=(32,64))
    n_filters, ix = r, 1
    for i in range(n_filters):
        f = filters[:, :, :, i]
        for j in range(c):
            ax = pyplot.subplot(n_filters, c, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(f[:,:, j], cmap='gray')
            ix += 1
    return fig

def show_filters(model):
    for layer in model.layers:
        if 'Conv' not in layer.name:
            continue
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)
        fig = show_filter(filters, filters.shape[3], filters.shape[2])
        fig.tight_layout()
        fig.savefig("filter_"+layer.name)

def show_layers(out, r, c):
    ix = 1
    for _ in range(r):
        for _ in range(c):
            ax = pyplot.subplot(r, c, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(out[0, :, :, ix-1], cmap='gray')
            ix+=1

def show_intermedium(model, img):
    l = model.layers[0](np.array([img,]))
    print([x.name for x in model.layers])
    fig = pyplot.figure(figsize=(16,8))
    show_layers(l,2,4)
    fig.tight_layout()
    fig.savefig("Conv1")

    l = model.layers[1](l)
    l = model.layers[2](l)
    fig = pyplot.figure(figsize=(16,16))
    show_layers(l,4,6)
    fig.tight_layout()
    fig.savefig("Conv2")
    
    l = model.layers[3](l)
    l = model.layers[4](l)
    fig = pyplot.figure(figsize=(16,16))
    show_layers(l,8,8)
    fig.canvas.draw()
    fig.tight_layout()
    pyplot.tight_layout()
    fig.savefig("Conv3")

def info_model(model, folder):
    print("Info about your model:")
    model.load_weights(CHECKPOINT_PATH)
    #model.summary()
    img, label = _get_img(folder, 0)
    pred = model.predict(np.array([img,]))
    print(pred)
    #show_intermedium(model, img)
    #show_filters(model)
    #pyplot.show()

def convert_tflite(model, folder):
    global BATCH_SIZE
    print("Converting model")
    BATCH_SIZE = 1
    model.load_weights(CHECKPOINT_PATH)
    train_data, val_data = get_data(folder)
    def representative_dataset():
        for x,val in train_data:
            yield [x]
    converter = tf.lite.TFLiteConverter.from_saved_model("model/1/")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()
    open("converted_model_quantized.tflite", "wb").write(tflite_model)
    del converter

    converter = tf.lite.TFLiteConverter.from_saved_model("model/1/")
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

def eval_model(folder):
    global BATCH_SIZE
    print("Evaluating model")
    BATCH_SIZE = 1
    train_data, val_data = get_data(folder)

    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]['index']
    out_details = interpreter.get_output_details()[0]['index']
    good = 0
    bad = 0
    t = 0.0
    for img, res in val_data:
        interpreter.set_tensor(in_details, img)
        start = time.time()
        interpreter.invoke()
        end = time.time()
        output_data = interpreter.get_tensor(out_details)
        if int(round(res.numpy()[0])) == int(round(output_data[0][0])):
            good += 1
        else:
            bad += 1
        t += end-start
    acc = good / (good+bad)
    t = t / (good+bad)
    print("TFlite: {} accuracy {}ms mean time".format(acc, t*1000.0))

    interpreter = tf.lite.Interpreter(model_path="converted_model_quantized.tflite")
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]['index']
    out_details = interpreter.get_output_details()[0]['index']
    
    good = 0
    bad = 0
    t = 0.0
    for img, res in val_data:
        interpreter.set_tensor(in_details, img)
        start = time.time()
        interpreter.invoke()
        end = time.time()
        output_data = interpreter.get_tensor(out_details)
        if int(round(res.numpy()[0])) == int(round(output_data[0][0])):
            good += 1
        else:
            bad += 1
        t += end-start
    acc = good / (good+bad)
    t = t / (good+bad)
    print("TFliteQuantized: {} accuracy {}ms mean time".format(acc, t*1000.0))

def main():
    args = parse_arguments()
    print("Loading Model")
    model = deepHotDog.DeepHotDog()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])
    model.build(input_shape=(None, HEIGHT, WIDTH, CHANNELS))

    if args.i:
        if args.f:
            info_model(model, args.f)
        else:
            print("I need pictures!!")
    if args.t:
        if args.f:
            train_model(model, args.f)
        else:
            print("I need pictures!!")
    if args.e:
        if args.f:
            eval_model(args.f)
        else:
            print("I need pictures!!")
    if args.c:
        if args.f:
            convert_tflite(model, args.f)
        else:
            print("I need pictures!!")

if __name__ == "__main__":
    main()