# ML_testbench_dashboard

A suite for automatic testing of ML models on Tensorflow Lite and ArmNN on Android Devices

# Requirements

You can install them with ```pip install -r requirements.txt```

# Usage

The testbench is a python script that accepts a yaml file with the configuration.
```
python3 testbench.py [yaml config file]
```
In the project we include an example of config file called test.yaml with a couple
of workload examples. We can config global settings which is the name of the output file like:
```
global:
  outputfile: name
```
After that we can select the workloads we want to execute as a list of options:
```
workloads:
  - name: name that will be shown in the dashboard
    model: path to the tflite model #Required
    tflite: #In case we want to test tflite models
      threads: [] list with the number of threads we want to try #Default 4
      options: [] list with the execution unit to use, cpu, gpu or nnapi #Default cpu
      loops: number of times to repeat the test #Default 10
    armnn: #In case we want to try armnn
      input_shape: [] list with the input shape #Required
      input_name: name of the input layer #Required
      output_name: name of the output layer #Required
      loops: number of times to repeat the test #Default 10
      concurrent: True or False depending if we want to use more than 1 cpu #Default False
      quantized: True or False if we want to quantize the input #Default False
      fp16: True or False if we want to use fp16 instead of fp32 #Default False
```
If you want to check the model to see the input shape and all the layers you can use,
Netron <https://github.com/lutzroeder/netron>.

Finally connect your ADB enabled Android device and run.