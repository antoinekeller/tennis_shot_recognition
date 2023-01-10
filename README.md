# tennis_shot_recognition

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



