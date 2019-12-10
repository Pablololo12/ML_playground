# Train

## Training

For training if you want to use two different classes you will need to change
the source code. On lines 20-21 ```CLASS_0``` and ```CLASS_1``` have the name
of the folders on our dataset.

Otherwise, the training and quantization is quite simple:

```bash
# This will train the model
python3 main.py -t -f [rootfolder of dataset]
# This will create two tflite files one quentized and one in floating point
python3 main.py -c -f [rootfolder of dataset]
# This will evaluate both of the tflite models
python3 main.py -e -f [rootfolder of dataset]
```

The script will create three folders:
* model - with the final model in .pb file
* checkpoint - in case of need to resume, in this case you will need to uncomment line 70 with model.load_weights()
* log - this is useful information for tensorboard

## Tensorboard

During the execution or at the end you can use tensorboard to get information
about the training. As soon as you see the log folder you can start a web server
with:
```tensorboard --logdir=log --host localhost --port 8088```

If you open the link that pops up you will get a board with all the information.

## Requirements

* Matplotlib
* numpy
* Tensorflow2.0
* scikit-learn