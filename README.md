# tennis_shot_recognition

## Movenet

To download the movenet_lightning_f16 neural network from Tensorflow, run :

```
wget -q -O movenet.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite
```

Note that multiple variants can be found to https://www.tensorflow.org/hub/tutorials/movenet if you want a different trade-off between precision and inference speed.

## Download data

To get tennis videos, you can simply download them from any youtube converter, e.g. https://en1.onlinevideoconverter.pro/11/

## extract_human_pose.py

<p>
<em>Movenet human pose estimation</em></br>
<img src="res/movenet_example.gif"  width="800" alt>
</p>

## Tennis shot annotation

To make your annotation, you can use the `annotator.py` file, e.g

```
$ python annotator.py dataset/nadal/nadal.mp4 
```

which will output a csv file, named `annotation_nadal.csv` containing something like this:

```
Shot,FrameId
serve,257
forehand,294
backhand,329
forehand,374
forehand,415
backhand,450
```

where each line corresponds to a shot at a specified frame.


