/*
 * Copyright (c) 2019, ARM Limited and Contributors
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package com.example.ishotdog;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class TFliteProf {

    private static final int MODEL_INPUT_SIZE = 224;
    private static final int BATCH_SIZE = 1;
    private static final int BYTES_PER_CHANNEL = Float.SIZE/Byte.SIZE;
    private static final int PIXEL_SIZE = 3;
    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;
    private static final int NUM_EXP = 1;
    //private static final String model_filename = "deepHotDog.tflite";
    private static final String model_filename = "deepHotDog_quant.tflite";

    public static void Inference (MainActivity main, Bitmap img, Handler handler, Options opt) {
        final MainActivity maine = main;
        final Handler handlerl = handler;
        String text = "";
        try {
            MappedByteBuffer model = getModelFile(maine);
            Interpreter.Options tfliteOptions = new Interpreter.Options();
            tfliteOptions.setNumThreads(4);
            tfliteOptions.setUseNNAPI(opt.getNNAPI());
            if (opt.getGPU())
                tfliteOptions.addDelegate(new GpuDelegate());
            if (opt.getNNAPIDEG())
                tfliteOptions.addDelegate(new NnApiDelegate());
            Log.d("Debug","Creating interpreter");
            Interpreter tfl = new Interpreter(model.asReadOnlyBuffer(), tfliteOptions);
            Log.d("Debug","Created interpreter");
            float result = recognize(img, tfl);

            if (result<0.5) {
                text = "Is a HotDog!!";
            } else {
                text = "Is not a hotdog :(";
            }
        } catch (IOException e) {
            e.printStackTrace();
            text = "Error opening label file OR tflite";
        }
        Message msg = new Message();
        msg.obj=text;
        handlerl.sendMessage(msg);
    }

    private static float recognize(Bitmap img, Interpreter model) {
        Bitmap image = Bitmap.createScaledBitmap(img, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, false);
        ByteBuffer byteBuffer = ByteBuffer
                .allocateDirect(BATCH_SIZE*
                        MODEL_INPUT_SIZE*
                        MODEL_INPUT_SIZE*
                        BYTES_PER_CHANNEL*
                        PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];
        byteBuffer.rewind();
        image.getPixels(intValues, 0, image.getWidth(), 0,0,image.getWidth(), image.getHeight());
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < MODEL_INPUT_SIZE; ++i) {
            for (int j = 0; j < MODEL_INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat(((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d("Debug","Timecost to put values into ByteBuffer: " + (endTime - startTime));

        float result [][] = new float[BATCH_SIZE][1];
        long acum = 0;
        Log.d("Debug", "Before Inference");
        for (int i=0; i<NUM_EXP; i++) {
            long start = SystemClock.uptimeMillis();
            Trace.beginSection("Inference");
            model.run(byteBuffer, result);
            Trace.endSection();
            long end = SystemClock.uptimeMillis();
            acum += end-start;
        }
        Log.d("Debug", "Mean Time to inference: "+String.format("%dms", acum/NUM_EXP));
        return result[0][0];
    }

    private static MappedByteBuffer getModelFile(Activity activity) throws IOException {
        String[] f = activity.getAssets().list("");
        for (String f1:f) {
            Log.d("Debug", f1);
        }
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(model_filename);
        Log.d("Debug", "File Opened");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
