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

import tensorflow as tf
import tensorflow.keras.layers as layers

class DeepHotDog(tf.keras.Model):
    def __init__(self):
        super(DeepHotDog, self).__init__()
        self.down1 = layers.Conv2D(8, kernel_size=3, activation='relu', name="Conv1")
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), name="Pooling1")
        self.down2 = layers.Conv2D(24, kernel_size=3, activation='relu', name="Conv2")
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), name="Pooling2")
        self.down3 = layers.Conv2D(64, kernel_size=3, activation='relu', name="Conv3")
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2), name="Pooling3")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu', name="Dense1")
        self.dense2 = layers.Dense(1, activation='sigmoid', name="Dense2")
    
    def call(self, x):
        x = self.down1(x)
        x = self.pool1(x)
        x = self.down2(x)
        x = self.pool2(x)
        x = self.down3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x