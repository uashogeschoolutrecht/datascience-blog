---
title: "TensorFlow examples in Python from RStudio"
author: "Marc A.T. Teunis, PhD" 
date: "2021-06-04 12:25:26"
output: 
  rmdformats::downcute:
    self_contained: TRUE
    number_sections: TRUE
    keep_md: TRUE
---



# Intro
I will try to run the Python script from RStudio, using the `{reticulate}` Python interface for R. The script we are going to run was derived [from](https://www.tensorflow.org/tutorials/quickstart/beginner) the beginners tutorial at the TensorFlow homepage. And can be found here: './py/tf_intro.py'

To get all files related to this tutorial, visit [this repo on Github:](https://github.com/uashogeschoolutrecht/python_from_rstudio) 

This short introduction uses Keras to:

    Build a neural network that classifies images.
    Train this neural network.
    And, finally, evaluate the accuracy of the model.

This document was build from RStudio using an RMarkdown literate programming script and was published to RStudio::CONNECT to host the html rendered version, you see here. The template used is ["rmdformats::downcute"](https://github.com/juba/rmdformats). Click this link for more details and more examples of other templates. The rendered version of this script is hosted [here](https://datascience.hu.nl/rsconnect/deep-learning-with-python-from-r/)

Download and install TensorFlow 2. Import TensorFlow into your Python environment. To keep things simple, I decided to install it in the base environment for now. I recommend to work with local virtual environments though.

Run the following one-time in the Terminal:


```bash
pip install tensorflow
pip install tensorflow_hub
pip install numpy
pip install pandas
```

# Packages and libraries
This loads the required R and Python packages to be able to run Python code in RStudio and to further process the model output in R. 

```r
library(tidyverse)
```

```
## -- Attaching packages --------------------------------------- tidyverse 1.3.1 --
```

```
## v ggplot2 3.3.3     v purrr   0.3.4
## v tibble  3.1.1     v dplyr   1.0.5
## v tidyr   1.1.3     v stringr 1.4.0
## v readr   1.4.0     v forcats 0.5.1
```

```
## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
## x dplyr::filter() masks stats::filter()
## x dplyr::lag()    masks stats::lag()
```

```r
library(reticulate)
```

Import python libraries in your R session

```r
reticulate::import("tensorflow")
```

```
## Module(tensorflow)
```


```python
import tensorflow as tf
```

# Load the data
Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

# Define model and Neural network
Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function for training:

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

# Get model metrics
For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.

```python
predictions = model(x_train[:1]).numpy()
predictions
```

```
## array([[ 0.37707955,  0.48998624, -0.60622954,  0.7012216 , -0.39343804,
##          0.37475258,  0.11868808,  0.2904308 , -0.27578825,  0.8989355 ]],
##       dtype=float32)
```

The tf.nn.softmax function converts these logits to "probabilities" for each class:

```python
tf.nn.softmax(predictions).numpy()
```

```
## array([[0.10832022, 0.12126746, 0.04051947, 0.14978993, 0.05012772,
##         0.10806846, 0.08365493, 0.09932955, 0.05638617, 0.18253607]],
##       dtype=float32)
```

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()
```

```
## 2.2249904
```

This loss is equal to the negative log probability of the true class: It is zero if the model is sure of the correct class.

This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.

# Model fit
The Model.fit method adjusts the model parameters to minimize the loss:     

```python
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
```

```
## Epoch 1/5
## 
##    1/1875 [..............................] - ETA: 16:53 - loss: 2.5265 - accuracy: 0.0625
##   29/1875 [..............................] - ETA: 3s - loss: 1.6381 - accuracy: 0.5119   
##   61/1875 [..............................] - ETA: 3s - loss: 1.2301 - accuracy: 0.6434
##   95/1875 [>.............................] - ETA: 2s - loss: 1.0094 - accuracy: 0.7046
##  124/1875 [>.............................] - ETA: 2s - loss: 0.8990 - accuracy: 0.7379
##  151/1875 [=>............................] - ETA: 2s - loss: 0.8256 - accuracy: 0.7587
##  185/1875 [=>............................] - ETA: 2s - loss: 0.7575 - accuracy: 0.7774
##  216/1875 [==>...........................] - ETA: 2s - loss: 0.7144 - accuracy: 0.7896
##  242/1875 [==>...........................] - ETA: 2s - loss: 0.6797 - accuracy: 0.7996
##  274/1875 [===>..........................] - ETA: 2s - loss: 0.6451 - accuracy: 0.8100
##  302/1875 [===>..........................] - ETA: 2s - loss: 0.6143 - accuracy: 0.8194
##  331/1875 [====>.........................] - ETA: 2s - loss: 0.5954 - accuracy: 0.8243
##  362/1875 [====>.........................] - ETA: 2s - loss: 0.5741 - accuracy: 0.8316
##  392/1875 [=====>........................] - ETA: 2s - loss: 0.5564 - accuracy: 0.8368
##  421/1875 [=====>........................] - ETA: 2s - loss: 0.5414 - accuracy: 0.8413
##  450/1875 [======>.......................] - ETA: 2s - loss: 0.5255 - accuracy: 0.8460
##  478/1875 [======>.......................] - ETA: 2s - loss: 0.5152 - accuracy: 0.8488
##  507/1875 [=======>......................] - ETA: 2s - loss: 0.5054 - accuracy: 0.8515
##  537/1875 [=======>......................] - ETA: 2s - loss: 0.4951 - accuracy: 0.8548
##  565/1875 [========>.....................] - ETA: 2s - loss: 0.4840 - accuracy: 0.8587
##  594/1875 [========>.....................] - ETA: 2s - loss: 0.4776 - accuracy: 0.8607
##  623/1875 [========>.....................] - ETA: 2s - loss: 0.4678 - accuracy: 0.8635
##  652/1875 [=========>....................] - ETA: 2s - loss: 0.4606 - accuracy: 0.8655
##  680/1875 [=========>....................] - ETA: 2s - loss: 0.4539 - accuracy: 0.8674
##  708/1875 [==========>...................] - ETA: 2s - loss: 0.4459 - accuracy: 0.8701
##  743/1875 [==========>...................] - ETA: 1s - loss: 0.4367 - accuracy: 0.8728
##  774/1875 [===========>..................] - ETA: 1s - loss: 0.4299 - accuracy: 0.8744
##  801/1875 [===========>..................] - ETA: 1s - loss: 0.4224 - accuracy: 0.8768
##  835/1875 [============>.................] - ETA: 1s - loss: 0.4144 - accuracy: 0.8790
##  865/1875 [============>.................] - ETA: 1s - loss: 0.4087 - accuracy: 0.8803
##  897/1875 [=============>................] - ETA: 1s - loss: 0.4019 - accuracy: 0.8822
##  936/1875 [=============>................] - ETA: 1s - loss: 0.3936 - accuracy: 0.8848
##  969/1875 [==============>...............] - ETA: 1s - loss: 0.3875 - accuracy: 0.8866
## 1002/1875 [===============>..............] - ETA: 1s - loss: 0.3827 - accuracy: 0.8883
## 1037/1875 [===============>..............] - ETA: 1s - loss: 0.3775 - accuracy: 0.8901
## 1069/1875 [================>.............] - ETA: 1s - loss: 0.3735 - accuracy: 0.8909
## 1098/1875 [================>.............] - ETA: 1s - loss: 0.3690 - accuracy: 0.8922
## 1132/1875 [=================>............] - ETA: 1s - loss: 0.3651 - accuracy: 0.8934
## 1162/1875 [=================>............] - ETA: 1s - loss: 0.3614 - accuracy: 0.8946
## 1191/1875 [==================>...........] - ETA: 1s - loss: 0.3576 - accuracy: 0.8959
## 1220/1875 [==================>...........] - ETA: 1s - loss: 0.3546 - accuracy: 0.8968
## 1252/1875 [===================>..........] - ETA: 1s - loss: 0.3514 - accuracy: 0.8976
## 1286/1875 [===================>..........] - ETA: 0s - loss: 0.3480 - accuracy: 0.8984
## 1320/1875 [====================>.........] - ETA: 0s - loss: 0.3446 - accuracy: 0.8994
## 1351/1875 [====================>.........] - ETA: 0s - loss: 0.3415 - accuracy: 0.9002
## 1379/1875 [=====================>........] - ETA: 0s - loss: 0.3387 - accuracy: 0.9010
## 1411/1875 [=====================>........] - ETA: 0s - loss: 0.3355 - accuracy: 0.9021
## 1441/1875 [======================>.......] - ETA: 0s - loss: 0.3330 - accuracy: 0.9028
## 1469/1875 [======================>.......] - ETA: 0s - loss: 0.3300 - accuracy: 0.9038
## 1504/1875 [=======================>......] - ETA: 0s - loss: 0.3267 - accuracy: 0.9048
## 1535/1875 [=======================>......] - ETA: 0s - loss: 0.3244 - accuracy: 0.9054
## 1565/1875 [========================>.....] - ETA: 0s - loss: 0.3220 - accuracy: 0.9062
## 1594/1875 [========================>.....] - ETA: 0s - loss: 0.3197 - accuracy: 0.9068
## 1626/1875 [=========================>....] - ETA: 0s - loss: 0.3168 - accuracy: 0.9076
## 1656/1875 [=========================>....] - ETA: 0s - loss: 0.3149 - accuracy: 0.9083
## 1689/1875 [==========================>...] - ETA: 0s - loss: 0.3120 - accuracy: 0.9091
## 1721/1875 [==========================>...] - ETA: 0s - loss: 0.3104 - accuracy: 0.9094
## 1753/1875 [===========================>..] - ETA: 0s - loss: 0.3078 - accuracy: 0.9102
## 1784/1875 [===========================>..] - ETA: 0s - loss: 0.3060 - accuracy: 0.9108
## 1817/1875 [============================>.] - ETA: 0s - loss: 0.3042 - accuracy: 0.9114
## 1846/1875 [============================>.] - ETA: 0s - loss: 0.3028 - accuracy: 0.9118
## 1875/1875 [==============================] - 4s 2ms/step - loss: 0.3006 - accuracy: 0.9124
## Epoch 2/5
## 
##    1/1875 [..............................] - ETA: 3s - loss: 0.2238 - accuracy: 0.9375
##   36/1875 [..............................] - ETA: 2s - loss: 0.1763 - accuracy: 0.9505
##   67/1875 [>.............................] - ETA: 2s - loss: 0.1789 - accuracy: 0.9510
##   97/1875 [>.............................] - ETA: 2s - loss: 0.1739 - accuracy: 0.9517
##  132/1875 [=>............................] - ETA: 2s - loss: 0.1793 - accuracy: 0.9496
##  163/1875 [=>............................] - ETA: 2s - loss: 0.1747 - accuracy: 0.9500
##  194/1875 [==>...........................] - ETA: 2s - loss: 0.1676 - accuracy: 0.9518
##  227/1875 [==>...........................] - ETA: 2s - loss: 0.1702 - accuracy: 0.9500
##  258/1875 [===>..........................] - ETA: 2s - loss: 0.1689 - accuracy: 0.9505
##  285/1875 [===>..........................] - ETA: 2s - loss: 0.1657 - accuracy: 0.9514
##  315/1875 [====>.........................] - ETA: 2s - loss: 0.1657 - accuracy: 0.9511
##  346/1875 [====>.........................] - ETA: 2s - loss: 0.1627 - accuracy: 0.9520
##  371/1875 [====>.........................] - ETA: 2s - loss: 0.1626 - accuracy: 0.9517
##  398/1875 [=====>........................] - ETA: 2s - loss: 0.1626 - accuracy: 0.9511
##  426/1875 [=====>........................] - ETA: 2s - loss: 0.1591 - accuracy: 0.9525
##  455/1875 [======>.......................] - ETA: 2s - loss: 0.1585 - accuracy: 0.9526
##  486/1875 [======>.......................] - ETA: 2s - loss: 0.1580 - accuracy: 0.9529
##  521/1875 [=======>......................] - ETA: 2s - loss: 0.1595 - accuracy: 0.9530
##  553/1875 [=======>......................] - ETA: 2s - loss: 0.1592 - accuracy: 0.9534
##  586/1875 [========>.....................] - ETA: 2s - loss: 0.1587 - accuracy: 0.9537
##  624/1875 [========>.....................] - ETA: 2s - loss: 0.1576 - accuracy: 0.9541
##  658/1875 [=========>....................] - ETA: 1s - loss: 0.1570 - accuracy: 0.9543
##  689/1875 [==========>...................] - ETA: 1s - loss: 0.1566 - accuracy: 0.9545
##  721/1875 [==========>...................] - ETA: 1s - loss: 0.1572 - accuracy: 0.9541
##  752/1875 [===========>..................] - ETA: 1s - loss: 0.1578 - accuracy: 0.9539
##  783/1875 [===========>..................] - ETA: 1s - loss: 0.1591 - accuracy: 0.9536
##  818/1875 [============>.................] - ETA: 1s - loss: 0.1596 - accuracy: 0.9533
##  846/1875 [============>.................] - ETA: 1s - loss: 0.1592 - accuracy: 0.9532
##  877/1875 [=============>................] - ETA: 1s - loss: 0.1576 - accuracy: 0.9539
##  909/1875 [=============>................] - ETA: 1s - loss: 0.1576 - accuracy: 0.9538
##  940/1875 [==============>...............] - ETA: 1s - loss: 0.1569 - accuracy: 0.9540
##  969/1875 [==============>...............] - ETA: 1s - loss: 0.1560 - accuracy: 0.9542
## 1001/1875 [===============>..............] - ETA: 1s - loss: 0.1551 - accuracy: 0.9545
## 1032/1875 [===============>..............] - ETA: 1s - loss: 0.1549 - accuracy: 0.9546
## 1061/1875 [===============>..............] - ETA: 1s - loss: 0.1553 - accuracy: 0.9543
## 1095/1875 [================>.............] - ETA: 1s - loss: 0.1546 - accuracy: 0.9544
## 1125/1875 [=================>............] - ETA: 1s - loss: 0.1540 - accuracy: 0.9547
## 1155/1875 [=================>............] - ETA: 1s - loss: 0.1534 - accuracy: 0.9548
## 1187/1875 [=================>............] - ETA: 1s - loss: 0.1534 - accuracy: 0.9549
## 1217/1875 [==================>...........] - ETA: 1s - loss: 0.1534 - accuracy: 0.9550
## 1250/1875 [===================>..........] - ETA: 1s - loss: 0.1537 - accuracy: 0.9550
## 1282/1875 [===================>..........] - ETA: 0s - loss: 0.1532 - accuracy: 0.9550
## 1313/1875 [====================>.........] - ETA: 0s - loss: 0.1531 - accuracy: 0.9552
## 1345/1875 [====================>.........] - ETA: 0s - loss: 0.1532 - accuracy: 0.9551
## 1376/1875 [=====================>........] - ETA: 0s - loss: 0.1527 - accuracy: 0.9552
## 1409/1875 [=====================>........] - ETA: 0s - loss: 0.1529 - accuracy: 0.9551
## 1440/1875 [======================>.......] - ETA: 0s - loss: 0.1520 - accuracy: 0.9554
## 1470/1875 [======================>.......] - ETA: 0s - loss: 0.1511 - accuracy: 0.9557
## 1502/1875 [=======================>......] - ETA: 0s - loss: 0.1511 - accuracy: 0.9557
## 1532/1875 [=======================>......] - ETA: 0s - loss: 0.1512 - accuracy: 0.9558
## 1562/1875 [=======================>......] - ETA: 0s - loss: 0.1507 - accuracy: 0.9557
## 1596/1875 [========================>.....] - ETA: 0s - loss: 0.1501 - accuracy: 0.9557
## 1628/1875 [=========================>....] - ETA: 0s - loss: 0.1494 - accuracy: 0.9559
## 1659/1875 [=========================>....] - ETA: 0s - loss: 0.1490 - accuracy: 0.9560
## 1692/1875 [==========================>...] - ETA: 0s - loss: 0.1483 - accuracy: 0.9562
## 1720/1875 [==========================>...] - ETA: 0s - loss: 0.1479 - accuracy: 0.9562
## 1750/1875 [===========================>..] - ETA: 0s - loss: 0.1475 - accuracy: 0.9563
## 1784/1875 [===========================>..] - ETA: 0s - loss: 0.1471 - accuracy: 0.9564
## 1815/1875 [============================>.] - ETA: 0s - loss: 0.1465 - accuracy: 0.9566
## 1846/1875 [============================>.] - ETA: 0s - loss: 0.1461 - accuracy: 0.9566
## 1875/1875 [==============================] - 3s 2ms/step - loss: 0.1464 - accuracy: 0.9563
## Epoch 3/5
## 
##    1/1875 [..............................] - ETA: 3s - loss: 0.1479 - accuracy: 0.9375
##   33/1875 [..............................] - ETA: 2s - loss: 0.1117 - accuracy: 0.9631
##   62/1875 [..............................] - ETA: 3s - loss: 0.1061 - accuracy: 0.9698
##   97/1875 [>.............................] - ETA: 2s - loss: 0.1126 - accuracy: 0.9662
##  131/1875 [=>............................] - ETA: 2s - loss: 0.1168 - accuracy: 0.9654
##  162/1875 [=>............................] - ETA: 2s - loss: 0.1179 - accuracy: 0.9651
##  193/1875 [==>...........................] - ETA: 2s - loss: 0.1180 - accuracy: 0.9649
##  224/1875 [==>...........................] - ETA: 2s - loss: 0.1165 - accuracy: 0.9654
##  256/1875 [===>..........................] - ETA: 2s - loss: 0.1144 - accuracy: 0.9658
##  290/1875 [===>..........................] - ETA: 2s - loss: 0.1139 - accuracy: 0.9658
##  322/1875 [====>.........................] - ETA: 2s - loss: 0.1137 - accuracy: 0.9659
##  351/1875 [====>.........................] - ETA: 2s - loss: 0.1165 - accuracy: 0.9658
##  383/1875 [=====>........................] - ETA: 2s - loss: 0.1176 - accuracy: 0.9656
##  415/1875 [=====>........................] - ETA: 2s - loss: 0.1172 - accuracy: 0.9650
##  446/1875 [======>.......................] - ETA: 2s - loss: 0.1173 - accuracy: 0.9652
##  478/1875 [======>.......................] - ETA: 2s - loss: 0.1154 - accuracy: 0.9653
##  510/1875 [=======>......................] - ETA: 2s - loss: 0.1150 - accuracy: 0.9649
##  541/1875 [=======>......................] - ETA: 2s - loss: 0.1147 - accuracy: 0.9647
##  570/1875 [========>.....................] - ETA: 2s - loss: 0.1151 - accuracy: 0.9649
##  599/1875 [========>.....................] - ETA: 2s - loss: 0.1147 - accuracy: 0.9650
##  627/1875 [=========>....................] - ETA: 2s - loss: 0.1144 - accuracy: 0.9651
##  658/1875 [=========>....................] - ETA: 1s - loss: 0.1135 - accuracy: 0.9651
##  694/1875 [==========>...................] - ETA: 1s - loss: 0.1131 - accuracy: 0.9650
##  726/1875 [==========>...................] - ETA: 1s - loss: 0.1138 - accuracy: 0.9649
##  756/1875 [===========>..................] - ETA: 1s - loss: 0.1144 - accuracy: 0.9651
##  789/1875 [===========>..................] - ETA: 1s - loss: 0.1139 - accuracy: 0.9651
##  822/1875 [============>.................] - ETA: 1s - loss: 0.1123 - accuracy: 0.9657
##  851/1875 [============>.................] - ETA: 1s - loss: 0.1119 - accuracy: 0.9657
##  884/1875 [=============>................] - ETA: 1s - loss: 0.1124 - accuracy: 0.9658
##  915/1875 [=============>................] - ETA: 1s - loss: 0.1118 - accuracy: 0.9660
##  944/1875 [==============>...............] - ETA: 1s - loss: 0.1111 - accuracy: 0.9662
##  977/1875 [==============>...............] - ETA: 1s - loss: 0.1107 - accuracy: 0.9664
## 1007/1875 [===============>..............] - ETA: 1s - loss: 0.1101 - accuracy: 0.9667
## 1037/1875 [===============>..............] - ETA: 1s - loss: 0.1104 - accuracy: 0.9667
## 1070/1875 [================>.............] - ETA: 1s - loss: 0.1101 - accuracy: 0.9667
## 1100/1875 [================>.............] - ETA: 1s - loss: 0.1096 - accuracy: 0.9669
## 1131/1875 [=================>............] - ETA: 1s - loss: 0.1094 - accuracy: 0.9670
## 1162/1875 [=================>............] - ETA: 1s - loss: 0.1089 - accuracy: 0.9673
## 1193/1875 [==================>...........] - ETA: 1s - loss: 0.1086 - accuracy: 0.9674
## 1224/1875 [==================>...........] - ETA: 1s - loss: 0.1089 - accuracy: 0.9673
## 1257/1875 [===================>..........] - ETA: 0s - loss: 0.1093 - accuracy: 0.9672
## 1289/1875 [===================>..........] - ETA: 0s - loss: 0.1092 - accuracy: 0.9671
## 1321/1875 [====================>.........] - ETA: 0s - loss: 0.1093 - accuracy: 0.9673
## 1354/1875 [====================>.........] - ETA: 0s - loss: 0.1089 - accuracy: 0.9674
## 1382/1875 [=====================>........] - ETA: 0s - loss: 0.1086 - accuracy: 0.9675
## 1408/1875 [=====================>........] - ETA: 0s - loss: 0.1087 - accuracy: 0.9676
## 1437/1875 [=====================>........] - ETA: 0s - loss: 0.1090 - accuracy: 0.9674
## 1466/1875 [======================>.......] - ETA: 0s - loss: 0.1094 - accuracy: 0.9674
## 1497/1875 [======================>.......] - ETA: 0s - loss: 0.1097 - accuracy: 0.9674
## 1528/1875 [=======================>......] - ETA: 0s - loss: 0.1096 - accuracy: 0.9674
## 1561/1875 [=======================>......] - ETA: 0s - loss: 0.1098 - accuracy: 0.9674
## 1593/1875 [========================>.....] - ETA: 0s - loss: 0.1096 - accuracy: 0.9675
## 1624/1875 [========================>.....] - ETA: 0s - loss: 0.1092 - accuracy: 0.9677
## 1646/1875 [=========================>....] - ETA: 0s - loss: 0.1090 - accuracy: 0.9677
## 1674/1875 [=========================>....] - ETA: 0s - loss: 0.1085 - accuracy: 0.9679
## 1704/1875 [==========================>...] - ETA: 0s - loss: 0.1081 - accuracy: 0.9681
## 1736/1875 [==========================>...] - ETA: 0s - loss: 0.1078 - accuracy: 0.9681
## 1769/1875 [===========================>..] - ETA: 0s - loss: 0.1082 - accuracy: 0.9680
## 1798/1875 [===========================>..] - ETA: 0s - loss: 0.1080 - accuracy: 0.9681
## 1830/1875 [============================>.] - ETA: 0s - loss: 0.1083 - accuracy: 0.9679
## 1859/1875 [============================>.] - ETA: 0s - loss: 0.1080 - accuracy: 0.9680
## 1875/1875 [==============================] - 3s 2ms/step - loss: 0.1078 - accuracy: 0.9681
## Epoch 4/5
## 
##    1/1875 [..............................] - ETA: 3s - loss: 0.0406 - accuracy: 1.0000
##   37/1875 [..............................] - ETA: 2s - loss: 0.0966 - accuracy: 0.9704
##   71/1875 [>.............................] - ETA: 2s - loss: 0.0930 - accuracy: 0.9714
##  106/1875 [>.............................] - ETA: 2s - loss: 0.0950 - accuracy: 0.9714
##  142/1875 [=>............................] - ETA: 2s - loss: 0.0926 - accuracy: 0.9725
##  175/1875 [=>............................] - ETA: 2s - loss: 0.0867 - accuracy: 0.9734
##  208/1875 [==>...........................] - ETA: 2s - loss: 0.0834 - accuracy: 0.9743
##  243/1875 [==>...........................] - ETA: 2s - loss: 0.0845 - accuracy: 0.9742
##  277/1875 [===>..........................] - ETA: 2s - loss: 0.0853 - accuracy: 0.9737
##  307/1875 [===>..........................] - ETA: 2s - loss: 0.0846 - accuracy: 0.9739
##  343/1875 [====>.........................] - ETA: 2s - loss: 0.0853 - accuracy: 0.9740
##  375/1875 [=====>........................] - ETA: 2s - loss: 0.0851 - accuracy: 0.9743
##  409/1875 [=====>........................] - ETA: 2s - loss: 0.0843 - accuracy: 0.9747
##  444/1875 [======>.......................] - ETA: 2s - loss: 0.0857 - accuracy: 0.9747
##  471/1875 [======>.......................] - ETA: 2s - loss: 0.0852 - accuracy: 0.9744
##  497/1875 [======>.......................] - ETA: 2s - loss: 0.0854 - accuracy: 0.9743
##  531/1875 [=======>......................] - ETA: 2s - loss: 0.0862 - accuracy: 0.9737
##  551/1875 [=======>......................] - ETA: 2s - loss: 0.0871 - accuracy: 0.9737
##  576/1875 [========>.....................] - ETA: 2s - loss: 0.0881 - accuracy: 0.9735
##  604/1875 [========>.....................] - ETA: 2s - loss: 0.0883 - accuracy: 0.9733
##  632/1875 [=========>....................] - ETA: 2s - loss: 0.0889 - accuracy: 0.9729
##  659/1875 [=========>....................] - ETA: 1s - loss: 0.0899 - accuracy: 0.9726
##  687/1875 [=========>....................] - ETA: 1s - loss: 0.0894 - accuracy: 0.9729
##  714/1875 [==========>...................] - ETA: 1s - loss: 0.0891 - accuracy: 0.9729
##  742/1875 [==========>...................] - ETA: 1s - loss: 0.0886 - accuracy: 0.9728
##  769/1875 [===========>..................] - ETA: 1s - loss: 0.0884 - accuracy: 0.9728
##  798/1875 [===========>..................] - ETA: 1s - loss: 0.0883 - accuracy: 0.9728
##  825/1875 [============>.................] - ETA: 1s - loss: 0.0883 - accuracy: 0.9728
##  854/1875 [============>.................] - ETA: 1s - loss: 0.0891 - accuracy: 0.9726
##  883/1875 [=============>................] - ETA: 1s - loss: 0.0893 - accuracy: 0.9725
##  912/1875 [=============>................] - ETA: 1s - loss: 0.0897 - accuracy: 0.9723
##  941/1875 [==============>...............] - ETA: 1s - loss: 0.0894 - accuracy: 0.9723
##  973/1875 [==============>...............] - ETA: 1s - loss: 0.0890 - accuracy: 0.9725
## 1006/1875 [===============>..............] - ETA: 1s - loss: 0.0894 - accuracy: 0.9724
## 1035/1875 [===============>..............] - ETA: 1s - loss: 0.0891 - accuracy: 0.9727
## 1069/1875 [================>.............] - ETA: 1s - loss: 0.0894 - accuracy: 0.9725
## 1099/1875 [================>.............] - ETA: 1s - loss: 0.0903 - accuracy: 0.9724
## 1130/1875 [=================>............] - ETA: 1s - loss: 0.0902 - accuracy: 0.9725
## 1164/1875 [=================>............] - ETA: 1s - loss: 0.0902 - accuracy: 0.9725
## 1195/1875 [==================>...........] - ETA: 1s - loss: 0.0901 - accuracy: 0.9726
## 1228/1875 [==================>...........] - ETA: 1s - loss: 0.0901 - accuracy: 0.9725
## 1263/1875 [===================>..........] - ETA: 1s - loss: 0.0897 - accuracy: 0.9728
## 1295/1875 [===================>..........] - ETA: 0s - loss: 0.0899 - accuracy: 0.9727
## 1326/1875 [====================>.........] - ETA: 0s - loss: 0.0899 - accuracy: 0.9727
## 1362/1875 [====================>.........] - ETA: 0s - loss: 0.0907 - accuracy: 0.9725
## 1397/1875 [=====================>........] - ETA: 0s - loss: 0.0903 - accuracy: 0.9726
## 1427/1875 [=====================>........] - ETA: 0s - loss: 0.0901 - accuracy: 0.9726
## 1464/1875 [======================>.......] - ETA: 0s - loss: 0.0896 - accuracy: 0.9727
## 1496/1875 [======================>.......] - ETA: 0s - loss: 0.0896 - accuracy: 0.9727
## 1528/1875 [=======================>......] - ETA: 0s - loss: 0.0898 - accuracy: 0.9726
## 1563/1875 [========================>.....] - ETA: 0s - loss: 0.0902 - accuracy: 0.9725
## 1596/1875 [========================>.....] - ETA: 0s - loss: 0.0901 - accuracy: 0.9726
## 1632/1875 [=========================>....] - ETA: 0s - loss: 0.0899 - accuracy: 0.9726
## 1665/1875 [=========================>....] - ETA: 0s - loss: 0.0900 - accuracy: 0.9726
## 1698/1875 [==========================>...] - ETA: 0s - loss: 0.0899 - accuracy: 0.9726
## 1732/1875 [==========================>...] - ETA: 0s - loss: 0.0899 - accuracy: 0.9726
## 1763/1875 [===========================>..] - ETA: 0s - loss: 0.0897 - accuracy: 0.9726
## 1793/1875 [===========================>..] - ETA: 0s - loss: 0.0897 - accuracy: 0.9728
## 1820/1875 [============================>.] - ETA: 0s - loss: 0.0897 - accuracy: 0.9728
## 1848/1875 [============================>.] - ETA: 0s - loss: 0.0898 - accuracy: 0.9727
## 1875/1875 [==============================] - 3s 2ms/step - loss: 0.0895 - accuracy: 0.9727
## Epoch 5/5
## 
##    1/1875 [..............................] - ETA: 1s - loss: 0.0164 - accuracy: 1.0000
##   33/1875 [..............................] - ETA: 2s - loss: 0.0851 - accuracy: 0.9716
##   67/1875 [>.............................] - ETA: 2s - loss: 0.0798 - accuracy: 0.9753
##  101/1875 [>.............................] - ETA: 2s - loss: 0.0714 - accuracy: 0.9780
##  135/1875 [=>............................] - ETA: 2s - loss: 0.0658 - accuracy: 0.9801
##  173/1875 [=>............................] - ETA: 2s - loss: 0.0642 - accuracy: 0.9801
##  208/1875 [==>...........................] - ETA: 2s - loss: 0.0639 - accuracy: 0.9803
##  242/1875 [==>...........................] - ETA: 2s - loss: 0.0636 - accuracy: 0.9802
##  276/1875 [===>..........................] - ETA: 2s - loss: 0.0671 - accuracy: 0.9792
##  306/1875 [===>..........................] - ETA: 2s - loss: 0.0666 - accuracy: 0.9794
##  340/1875 [====>.........................] - ETA: 2s - loss: 0.0661 - accuracy: 0.9797
##  375/1875 [=====>........................] - ETA: 2s - loss: 0.0666 - accuracy: 0.9796
##  413/1875 [=====>........................] - ETA: 2s - loss: 0.0665 - accuracy: 0.9790
##  445/1875 [======>.......................] - ETA: 2s - loss: 0.0681 - accuracy: 0.9777
##  479/1875 [======>.......................] - ETA: 2s - loss: 0.0686 - accuracy: 0.9775
##  517/1875 [=======>......................] - ETA: 2s - loss: 0.0689 - accuracy: 0.9775
##  550/1875 [=======>......................] - ETA: 1s - loss: 0.0693 - accuracy: 0.9774
##  583/1875 [========>.....................] - ETA: 1s - loss: 0.0695 - accuracy: 0.9775
##  619/1875 [========>.....................] - ETA: 1s - loss: 0.0694 - accuracy: 0.9779
##  652/1875 [=========>....................] - ETA: 1s - loss: 0.0684 - accuracy: 0.9782
##  683/1875 [=========>....................] - ETA: 1s - loss: 0.0689 - accuracy: 0.9779
##  721/1875 [==========>...................] - ETA: 1s - loss: 0.0687 - accuracy: 0.9780
##  753/1875 [===========>..................] - ETA: 1s - loss: 0.0688 - accuracy: 0.9781
##  786/1875 [===========>..................] - ETA: 1s - loss: 0.0687 - accuracy: 0.9782
##  824/1875 [============>.................] - ETA: 1s - loss: 0.0694 - accuracy: 0.9782
##  853/1875 [============>.................] - ETA: 1s - loss: 0.0688 - accuracy: 0.9785
##  883/1875 [=============>................] - ETA: 1s - loss: 0.0682 - accuracy: 0.9786
##  917/1875 [=============>................] - ETA: 1s - loss: 0.0680 - accuracy: 0.9786
##  949/1875 [==============>...............] - ETA: 1s - loss: 0.0689 - accuracy: 0.9782
##  983/1875 [==============>...............] - ETA: 1s - loss: 0.0691 - accuracy: 0.9780
## 1019/1875 [===============>..............] - ETA: 1s - loss: 0.0690 - accuracy: 0.9779
## 1053/1875 [===============>..............] - ETA: 1s - loss: 0.0691 - accuracy: 0.9778
## 1086/1875 [================>.............] - ETA: 1s - loss: 0.0690 - accuracy: 0.9777
## 1119/1875 [================>.............] - ETA: 1s - loss: 0.0690 - accuracy: 0.9777
## 1149/1875 [=================>............] - ETA: 1s - loss: 0.0697 - accuracy: 0.9778
## 1181/1875 [=================>............] - ETA: 1s - loss: 0.0713 - accuracy: 0.9774
## 1214/1875 [==================>...........] - ETA: 0s - loss: 0.0722 - accuracy: 0.9771
## 1247/1875 [==================>...........] - ETA: 0s - loss: 0.0728 - accuracy: 0.9769
## 1277/1875 [===================>..........] - ETA: 0s - loss: 0.0733 - accuracy: 0.9768
## 1310/1875 [===================>..........] - ETA: 0s - loss: 0.0730 - accuracy: 0.9769
## 1338/1875 [====================>.........] - ETA: 0s - loss: 0.0733 - accuracy: 0.9769
## 1368/1875 [====================>.........] - ETA: 0s - loss: 0.0729 - accuracy: 0.9771
## 1401/1875 [=====================>........] - ETA: 0s - loss: 0.0734 - accuracy: 0.9769
## 1431/1875 [=====================>........] - ETA: 0s - loss: 0.0736 - accuracy: 0.9769
## 1463/1875 [======================>.......] - ETA: 0s - loss: 0.0740 - accuracy: 0.9768
## 1494/1875 [======================>.......] - ETA: 0s - loss: 0.0742 - accuracy: 0.9767
## 1525/1875 [=======================>......] - ETA: 0s - loss: 0.0743 - accuracy: 0.9767
## 1558/1875 [=======================>......] - ETA: 0s - loss: 0.0743 - accuracy: 0.9768
## 1589/1875 [========================>.....] - ETA: 0s - loss: 0.0744 - accuracy: 0.9768
## 1622/1875 [========================>.....] - ETA: 0s - loss: 0.0744 - accuracy: 0.9768
## 1652/1875 [=========================>....] - ETA: 0s - loss: 0.0743 - accuracy: 0.9768
## 1678/1875 [=========================>....] - ETA: 0s - loss: 0.0743 - accuracy: 0.9768
## 1708/1875 [==========================>...] - ETA: 0s - loss: 0.0745 - accuracy: 0.9767
## 1739/1875 [==========================>...] - ETA: 0s - loss: 0.0749 - accuracy: 0.9766
## 1769/1875 [===========================>..] - ETA: 0s - loss: 0.0748 - accuracy: 0.9767
## 1805/1875 [===========================>..] - ETA: 0s - loss: 0.0750 - accuracy: 0.9767
## 1836/1875 [============================>.] - ETA: 0s - loss: 0.0752 - accuracy: 0.9767
## 1866/1875 [============================>.] - ETA: 0s - loss: 0.0751 - accuracy: 0.9767
## 1875/1875 [==============================] - 3s 2ms/step - loss: 0.0752 - accuracy: 0.9767
## <tensorflow.python.keras.callbacks.History object at 0x0000000039869438>
```

```python
model.evaluate(x_test,  y_test, verbose=2)
```

```
## 313/313 - 0s - loss: 0.0676 - accuracy: 0.9786
## [0.06759259104728699, 0.978600025177002]
```

# Evaluate
The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".

The image classifier is now trained to ~98% accuracy on this dataset. To learn more, read the TensorFlow tutorials.

If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:

```python
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])
```

```
## <tf.Tensor: shape=(5, 10), dtype=float32, numpy=
## array([[6.10386541e-09, 5.35523359e-09, 7.71399427e-05, 4.65752491e-05,
##         1.01343265e-11, 6.57291878e-07, 2.31564361e-13, 9.99875307e-01,
##         8.92304968e-08, 1.29458684e-07],
##        [3.90865154e-08, 2.53726932e-04, 9.99726355e-01, 1.51731883e-05,
##         7.30199575e-17, 2.67177029e-06, 1.41895484e-07, 1.97020577e-11,
##         1.95178222e-06, 2.39910110e-12],
##        [1.74124054e-07, 9.99440968e-01, 3.06388065e-05, 6.89203898e-06,
##         1.02222417e-04, 7.60997136e-06, 2.46908057e-05, 2.67468859e-04,
##         1.16548770e-04, 2.69115549e-06],
##        [9.99347031e-01, 1.44142724e-08, 5.31349040e-04, 1.52490287e-07,
##         5.35717959e-07, 3.91729827e-06, 1.00148784e-04, 8.20432376e-07,
##         1.67225167e-07, 1.59722222e-05],
##        [3.11061945e-06, 2.28186262e-10, 1.50921851e-05, 6.94476583e-08,
##         9.97447610e-01, 1.57998755e-07, 2.41655016e-05, 2.35697051e-04,
##         1.57465126e-06, 2.27258005e-03]], dtype=float32)>
```

# Get the results into R

To work with the results from the previous Python code, To access Python objects from R in RStudio you can use the `py` R object. Subsetting to an individual Python object is done using the `$` R operator. On the other hand R objects can be accessed from Python via the `r` object in your Python environment. Subsetting on the R object level can be achieved using a `.` (dot)). Below I show you examples for both cases:

## Accessing Python objects from R

```r
py$x_test %>% as_tibble()
```

```
## # A tibble: 10,000 x 784
##       V1    V2    V3    V4    V5    V6    V7    V8    V9   V10   V11   V12   V13
##    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
##  1     0     0     0     0     0     0     0     0     0     0     0     0     0
##  2     0     0     0     0     0     0     0     0     0     0     0     0     0
##  3     0     0     0     0     0     0     0     0     0     0     0     0     0
##  4     0     0     0     0     0     0     0     0     0     0     0     0     0
##  5     0     0     0     0     0     0     0     0     0     0     0     0     0
##  6     0     0     0     0     0     0     0     0     0     0     0     0     0
##  7     0     0     0     0     0     0     0     0     0     0     0     0     0
##  8     0     0     0     0     0     0     0     0     0     0     0     0     0
##  9     0     0     0     0     0     0     0     0     0     0     0     0     0
## 10     0     0     0     0     0     0     0     0     0     0     0     0     0
## # ... with 9,990 more rows, and 771 more variables: V14 <dbl>, V15 <dbl>,
## #   V16 <dbl>, V17 <dbl>, V18 <dbl>, V19 <dbl>, V20 <dbl>, V21 <dbl>,
## #   V22 <dbl>, V23 <dbl>, V24 <dbl>, V25 <dbl>, V26 <dbl>, V27 <dbl>,
## #   V28 <dbl>, V29 <dbl>, V30 <dbl>, V31 <dbl>, V32 <dbl>, V33 <dbl>,
## #   V34 <dbl>, V35 <dbl>, V36 <dbl>, V37 <dbl>, V38 <dbl>, V39 <dbl>,
## #   V40 <dbl>, V41 <dbl>, V42 <dbl>, V43 <dbl>, V44 <dbl>, V45 <dbl>,
## #   V46 <dbl>, V47 <dbl>, V48 <dbl>, V49 <dbl>, V50 <dbl>, V51 <dbl>,
## #   V52 <dbl>, V53 <dbl>, V54 <dbl>, V55 <dbl>, V56 <dbl>, V57 <dbl>,
## #   V58 <dbl>, V59 <dbl>, V60 <dbl>, V61 <dbl>, V62 <dbl>, V63 <dbl>,
## #   V64 <dbl>, V65 <dbl>, V66 <dbl>, V67 <dbl>, V68 <dbl>, V69 <dbl>,
## #   V70 <dbl>, V71 <dbl>, V72 <dbl>, V73 <dbl>, V74 <dbl>, V75 <dbl>,
## #   V76 <dbl>, V77 <dbl>, V78 <dbl>, V79 <dbl>, V80 <dbl>, V81 <dbl>,
## #   V82 <dbl>, V83 <dbl>, V84 <dbl>, V85 <dbl>, V86 <dbl>, V87 <dbl>,
## #   V88 <dbl>, V89 <dbl>, V90 <dbl>, V91 <dbl>, V92 <dbl>, V93 <dbl>,
## #   V94 <dbl>, V95 <dbl>, V96 <dbl>, V97 <dbl>, V98 <dbl>, V99 <dbl>,
## #   V100 <dbl>, V101 <dbl>, V102 <dbl>, V103 <dbl>, V104 <dbl>, V105 <dbl>,
## #   V106 <dbl>, V107 <dbl>, V108 <dbl>, V109 <dbl>, V110 <dbl>, V111 <dbl>,
## #   V112 <dbl>, V113 <dbl>, ...
```

```r
py$predictions %>% as_tibble()
```

```
## Warning: The `x` argument of `as_tibble.matrix()` must have unique column names if `.name_repair` is omitted as of tibble 2.0.0.
## Using compatibility `.name_repair`.
```

```
## # A tibble: 1 x 10
##      V1    V2     V3    V4     V5    V6    V7    V8     V9   V10
##   <dbl> <dbl>  <dbl> <dbl>  <dbl> <dbl> <dbl> <dbl>  <dbl> <dbl>
## 1 0.377 0.490 -0.606 0.701 -0.393 0.375 0.119 0.290 -0.276 0.899
```

## Accessing R objects from Python
To do this we first need an R object. So far we did not create anything in our R environment, we just loaded a bunch of R packages. Let's stick to MNIST. There is also an R implementation of Keras called the `{keras}` R package. This R `{keras}` package has the MNIST data set also build in. Let's load that dataset as an R object and then access that data set from Python.

The code below is not a very R way of doing things. In R you should try to keep things together in a list or a dataframe, in stead of having these loose vectors flying around in different places. There are better ways to do this, but for now I leave it like this so you can really follow what's going on. More compact code is often a bit harder to read. 

```r
library(keras)
mnist_r <- dataset_mnist()
x_train <- mnist_r$train$x
y_train <- mnist_r$train$y
x_test <- mnist_r$test$x
y_test <- mnist_r$test$y

## show one object
x_train %>% as_tibble()
```

```
## # A tibble: 60,000 x 784
##       V1    V2    V3    V4    V5    V6    V7    V8    V9   V10   V11   V12   V13
##    <int> <int> <int> <int> <int> <int> <int> <int> <int> <int> <int> <int> <int>
##  1     0     0     0     0     0     0     0     0     0     0     0     0     0
##  2     0     0     0     0     0     0     0     0     0     0     0     0     0
##  3     0     0     0     0     0     0     0     0     0     0     0     0     0
##  4     0     0     0     0     0     0     0     0     0     0     0     0     0
##  5     0     0     0     0     0     0     0     0     0     0     0     0     0
##  6     0     0     0     0     0     0     0     0     0     0     0     0     0
##  7     0     0     0     0     0     0     0     0     0     0     0     0     0
##  8     0     0     0     0     0     0     0     0     0     0     0     0     0
##  9     0     0     0     0     0     0     0     0     0     0     0     0     0
## 10     0     0     0     0     0     0     0     0     0     0     0     0     0
## # ... with 59,990 more rows, and 771 more variables: V14 <int>, V15 <int>,
## #   V16 <int>, V17 <int>, V18 <int>, V19 <int>, V20 <int>, V21 <int>,
## #   V22 <int>, V23 <int>, V24 <int>, V25 <int>, V26 <int>, V27 <int>,
## #   V28 <int>, V29 <int>, V30 <int>, V31 <int>, V32 <int>, V33 <int>,
## #   V34 <int>, V35 <int>, V36 <int>, V37 <int>, V38 <int>, V39 <int>,
## #   V40 <int>, V41 <int>, V42 <int>, V43 <int>, V44 <int>, V45 <int>,
## #   V46 <int>, V47 <int>, V48 <int>, V49 <int>, V50 <int>, V51 <int>,
## #   V52 <int>, V53 <int>, V54 <int>, V55 <int>, V56 <int>, V57 <int>,
## #   V58 <int>, V59 <int>, V60 <int>, V61 <int>, V62 <int>, V63 <int>,
## #   V64 <int>, V65 <int>, V66 <int>, V67 <int>, V68 <int>, V69 <int>,
## #   V70 <int>, V71 <int>, V72 <int>, V73 <int>, V74 <int>, V75 <int>,
## #   V76 <int>, V77 <int>, V78 <int>, V79 <int>, V80 <int>, V81 <int>,
## #   V82 <int>, V83 <int>, V84 <int>, V85 <int>, V86 <int>, V87 <int>,
## #   V88 <int>, V89 <int>, V90 <int>, V91 <int>, V92 <int>, V93 <int>,
## #   V94 <int>, V95 <int>, V96 <int>, V97 <int>, V98 <int>, V99 <int>,
## #   V100 <int>, V101 <int>, V102 <int>, V103 <int>, V104 <int>, V105 <int>,
## #   V106 <int>, V107 <int>, V108 <int>, V109 <int>, V110 <int>, V111 <int>,
## #   V112 <int>, V113 <int>, ...
```

Now we will access one of the above R objects from Python

```python
r.x_train
```

```
## array([[[0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         ...,
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0]],
## 
##        [[0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         ...,
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0]],
## 
##        [[0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         ...,
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0]],
## 
##        ...,
## 
##        [[0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         ...,
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0]],
## 
##        [[0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         ...,
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0]],
## 
##        [[0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         ...,
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0],
##         [0, 0, 0, ..., 0, 0, 0]]])
```

# Plotting the MNIST hand written digits in R
Now that we know how to access the MNIST data from R, we can leverage R's visualization powers, without the need to load the data into R again (we have it already in our Python environment from the code we ran in the first steps of this demo). 
Below, I show how to access the training data in R by using the `py` environment. You can also interactively explore all R and Python objects in RStudio:


```r
knitr::include_graphics(
  here::here(
    "img",
    "python_env.png"
  )
)
```

![]("/blog/assets/python_env.png" width="1920")

```r
knitr::include_graphics(
  here::here(
    "img",
    "r_env.png"
  )
)
```

<img src="/blog/assets/r_env.png" width="1920" />

## Do the actual plotting
First, we use base-R to plot the first 36 digits in the MNIST train dataset.

```r
mnist <- py$mnist
x_train <- py$x_train
y_train <- py$y_train
  
# visualize the digits
par(mfcol=c(6,6))
par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')
for (idx in 1:36) { 
     im <- x_train[idx,,]
    im <- t(apply(im, 2, rev)) 
    image(1:28, 1:28, im, col=gray((0:255)/255), 
          xaxt='n', main=paste(y_train[idx]))
}
```

![](convulation_networks_tensorflow_inPython_from_R_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

# Plotting using {ggplot2}
R is famous for it's plotting capabilities. I prefer the grammar of graphics above the other two systems in R (`{base}` and `{lattice}`). The grammar of graphics has been implemented in the `{ggplot2}` R package. Here I show how to plot 1 digit randomly sampled from the MNIST train data.

```r
set.seed(1234)
py$x_train[sample(1:100,1), 1:28, 1:28] %>%
                as_tibble() %>% 
                rownames_to_column(var = 'y') %>% 
                gather(x, val, V1:V28) %>%
                mutate(x = str_replace(x, 'V', '')) %>% 
                mutate(x = as.numeric(x),
                       y = as.numeric(y)) %>% 
                mutate(y = 28-y) %>%
                ggplot(aes(x, y))+
                geom_tile(aes(fill = val+1))+
                coord_fixed()+
                theme_void()+
                theme(legend.position="none")
```

![](convulation_networks_tensorflow_inPython_from_R_files/figure-html/unnamed-chunk-16-1.png)<!-- -->

# Source the complete python script
You can also source the complete python script as showed below. This will execute all the Python commands showed in this demo. Te individual Python objects will be avaialbe to you in R via `py$<python_object_name>` and in Python in the Python REPL and via `r.<python_object_name>`. This is all levraged in RStudio by the [`{reticulate}` R package](https://rstudio.github.io/reticulate/) 


```r
reticulate::source_python(
  here::here(
    "py",
    "tf_intro.py"
  )
)
```

