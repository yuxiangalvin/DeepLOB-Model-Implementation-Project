
## DeepLOB-Model-Implementation-Project

Welcome to my project page! It's time to dive Deep.

This is a replicate project to develop the model raised up in the paper DeepLOB - Deep Convolutional Neural Networks. 
The original paper's authors are Zihao Zhang, Stefan Zohren, Stephen Roberts. The paper is linked here: https://arxiv.org/pdf/1808.03668.pdf

### Project Information
#### University: Northwestern University

#### Professor: Prof. Han Liu

#### Project Member & Contact Information:
  
  * Yuxiang(Alvin) Chen   yuxiangchen2021 at u.northwestern.edu

#### GitHub Repository:
  Here is my [GitHub Repository](https://github.com/yuxiangalvin/DeepLOB-Model-Implementation-Project)
  This repo contains some codes and outputs of my implementation of DeepLOB model.
  
### Motivation:

Deep Learning's application in Finance has always been one of the most complicated research area for Deep Learning. While reading various papers that focus on Deep Learning methods on Quantitative Finance applications, this paper about [DeepLOB - Deep Convolutional Neural Networks](https://arxiv.org/pdf/1808.03668.pdf) catches my attention.

Nowadays, most companies in Quantitative Finance field uses limit orderbook data to conduct all kinds of analysis. It provides much more information than traditional one point data. High frequency limit orderbook data is essential for companies which conduct high frequency trading and the importance has been growing at an extremely fast speed. As an individal who is very passionate about machine learning's applications in finance data of Level II or above, I would like to fully understand the DeepLOB model and the authors' intensions behind each design component. At the same time, I would like to further enhance my deep learning application skills. Thus, I conducted this replicate project.


### Model Details

The model takes in the limit orderbook data of one specific stock as input and conduct a 3-class prediction for the stock's mid-price movement. The three classes are 'Down', 'Hold' and 'Up'. There has been previous researches which focus on limit orderbook data. However, most of them applied static feature extract method that mainly based on domain expertise and conventions. These static methods include Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA), etc. 

The DeepLOB model intead applies a dynamic feature ectraction approach through applying Deep Learning architecture (CNN + Inception Module). It also uses an additional LSTM to capture additional dependence that is not captured by the CNN + Inception Module.

#### Inputs

##### Raw Data
The DeepLOB model takes limit orderbook data as inputs, specifically, at each time point, it takes in a limit orderbook snippet - the lowest 10 ask levels and the highest 10 bid levels. Each level has one price data and one size data. Thus, at each time point, there are 40 numbers. Below is am example of how the orderbook looks like at one time point (10 levels are shown here in this example)

![LIMIT ORDERBOOK EXAMPLE](./src/images/limit_orderbook_example.png)


The authors of the model use a lookback period of 100 timesteps at each time step. Thus at each time step, the input matrix has a dimension of 100 x 40.

Thus the input size of the model is N x 100 x 40 x 1 (N is the number of timesteps used as input)

The paper authors used two different datasets: FI-2010 and London Stock Exchange (LSE) limit orderbook data.

##### FI-2010 dataset

* FI-2010 is a public benchmark dataset of HFT LOB data and extracted time series data for five stocks from the Nasdaq Nordic stock market for a time period of 10 consecutive days. 
* The timestep distance between two data points are in average < 1 second. 
* The dataset is pre-normalized using z-score normalization.

##### LSE dataset

* LSE dataset is not a publicly accessible dataset. 
* The stocks involved has much higher liqudity than the 5 stocks in FI-2010 dataset
* The timestep distance between two data points are samller and in average 0.192 seocnd
* The dataset is not pre-normalized.

##### Data Labeling
Following quantities are calculated using corresponding equations & labels are generated.

|  Dataset  | mid-price | previous k timesteps avg mid-price | future k timesteps avg mid-price| move pct | label
| FI-2010 |   | - | ------- |  |  |
| LSE | ---  | - | ------- |  |  |


##### Data Normalization

* FI-2010 dataset is pre-noralized 
* LSE dataset: Each trading day's price and size at each level is normalized using the previous 5 trading days' price and size separately.


#### Model Structure

Here I will use the original pictures used in the original paper with my annotations to present the model structure.

![WHOLE MODEL STRUCTURE](./src/images/whole_model_structure.png)

The model starts with 1 CNN block with 3 sub parts. 

##### CNN Block Design

There are three points that worths noticing in the CNN block design.

1. The design of 1x2 filter and 1x2 stride at the beginning the 1st sub part is used to capture one important nature of the input data. One of the dimentions of the input data is 40 (price and size at 20 levels of order book). Since the data is ordered as price, size alternatively. This design keeps the first element of the horizontal filter only looking at prices and the second element of the filter only looking at sizes. This design takes the nature of the data into account and thus makes the 16 different feature maps generated from 16 different filters more representative.

2. The design of 4x1 filter in the 2nd layer of 1st subpart capture local interactions amongst data over four time steps.

3. The further layers keep exploring boarder interactions.

##### Inception Module

Following the CNN block is an Inception Module. The Inception Module is more powerful than a common CNN block becasue it allows to use multiple types of filter size, instead of being restricted to a single filter size. The specific structure of the Inception Module is shown below in the figure.

![INCEPTION MODULE STRUCTURE](./src/images/inception_module_structure.png)

As the structure figure shows, this specific Inception Module contains three parallel processes. This allows the module to capture dynamic behaviors over multiple timescales. An 1 x 1 Conv layer is used in every path. This idea is form the Network-in-Network approach proposed in a [2014 paper](https://arxiv.org/pdf/1312.4400v3.pdf). Instead of applying a simple convolution to the data, the Network-in-Network method uses a small neural network to capture the non-linear properties of the data.

##### LSTM & FC

A LSTM layer with 64 LSTM unities is used after the CNN + Inception Module part in order to capture additioanl time dependencies.

A fully connected layer is used to map the 64 outputs from LSTM units to size 3 (one hot encoding of the 3 categories)


#### My Replicate Codes

##### Import Libraries
```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, concatenate, LSTM, Reshape, Dense
from keras.callbacks import EarlyStopping

import pandas_market_calendars as mcal
```

##### Data Normalization & Labeling

I conducted experiment on JNJ (Johnson & Johnson) 2020 January limit orderbook dataset.
* About 160000 - 200000 timesteps (data points) per trading day
* Average timestep distance 0.15 second

Here is the code 



##### Hyper-parameters
```python
#Input param
lookback_timestep = 100
feature_num = 40

#Conv param
conv_filter_num = 16

#Inception module param
inception_num = 32

#LSTM param
LSTM_num = 64

#Activation param
leaky_relu_alpha = 0.01

#Training params
loss = 'categorical_crossentropy'
learning_rate = 0.01
adam_epsilon = 1
optimizer = Adam(lr=learning_rate, epsilon=1)
batch_size = 32

#Training stopping Criteria
metrics = ['accuracy']
#stop training when validation accuracy does not improve for 20 epochs
stop_epoch_num = 20

#max epoch num is not specified in paper, use an arbitrary large number 10000
num_epoch = 10000
```

##### Model Inititation

```python
def initiate_DeepLOB_model(lookback_timestep, feature_num, conv_filter_num, inception_num, LSTM_num, leaky_relu_alpha,
                          loss, optimizer, metrics):
    
    input_tensor = Input(shape=(lookback_timestep, feature_num, 1))
    
    # Conv block1
    print(input_tensor.shape)
    conv_layer1 = Conv2D(conv_filter_num, (1,2), strides=(1, 2))(input_tensor)
    print(conv_layer1.shape)
    conv_layer1 =LeakyReLU(alpha=leaky_relu_alpha)(conv_layer1)
    print(conv_layer1.shape)
    conv_layer1 = Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer1)
    conv_first1 = LeakyReLU(alpha=leaky_relu_alpha)(conv_layer1)
    print(conv_layer1.shape)
    conv_layer1 = Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer1)
    conv_layer1 = LeakyReLU(alpha=leaky_relu_alpha)(conv_layer1)
    print(conv_layer1.shape)

    # Conv block2
    conv_layer2 = Conv2D(conv_filter_num, (1,2), strides=(1, 2))(conv_layer1)
    conv_layer2 = LeakyReLU(alpha=leaky_relu_alpha)(conv_layer2)
    print(conv_layer2.shape)
    conv_layer2 = Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer2)
    conv_layer2 = LeakyReLU(alpha=leaky_relu_alpha)(conv_layer2)
    print(conv_layer2.shape)
    conv_layer2 = Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer2)
    conv_layer2 = LeakyReLU(alpha=leaky_relu_alpha)(conv_layer2)
    print(conv_layer2.shape)

    # Conv block3
    conv_layer3 = Conv2D(conv_filter_num, (1,10))(conv_layer2)
    conv_layer3 = LeakyReLU(alpha=leaky_relu_alpha)(conv_layer3)
    print(conv_layer3.shape)
    conv_layer3 = Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer3)
    conv_layer3 = LeakyReLU(alpha=leaky_relu_alpha)(conv_layer3)
    print(conv_layer3.shape)
    conv_layer3 = Conv2D(conv_filter_num, (4,1), padding='same')(conv_layer3)
    conv_layer3 = LeakyReLU(alpha=leaky_relu_alpha)(conv_layer3)
    print(conv_layer3.shape)
    
    # Inception module
    inception_module1 = Conv2D(inception_num, (1,1), padding='same')(conv_layer3)
    inception_module1 = LeakyReLU(alpha=leaky_relu_alpha)(inception_module1)
    print(inception_module1.shape)
    inception_module1 = Conv2D(inception_num, (3,1), padding='same')(inception_module1)
    inception_module1 = LeakyReLU(alpha=leaky_relu_alpha)(inception_module1)
    print(inception_module1.shape)

    inception_module2 = Conv2D(inception_num, (1,1), padding='same')(conv_layer3)
    inception_module2 = LeakyReLU(alpha=leaky_relu_alpha)(inception_module2)
    print(inception_module2.shape)
    inception_module2 = Conv2D(inception_num, (5,1), padding='same')(inception_module2)
    inception_module2 = LeakyReLU(alpha=leaky_relu_alpha)(inception_module2)
    print(inception_module2.shape)

    inception_module3 = MaxPooling2D((3,1), strides=(1,1), padding='same')(conv_layer3)
    print(inception_module3.shape)
    inception_module3 = Conv2D(inception_num, (1,1), padding='same')(inception_module3)
    print(inception_module3.shape)
    inception_module3 = LeakyReLU(alpha=leaky_relu_alpha)(inception_module3)
    print(inception_module3.shape)
    
    inception_module_final = concatenate([inception_module1, inception_module2, inception_module3], axis=3)
    print(inception_module_final.shape)
    inception_module_final = Reshape((inception_module_final.shape[1], inception_module_final.shape[3]))(inception_module_final)
    print(inception_module_final.shape)

    # LSTM
    LSTM_output = LSTM(LSTM_num)(inception_module_final)
    print(LSTM_output.shape)

    # Fully Connected Layer with softmax activation function for output
    model_output = Dense(3, activation='softmax')(LSTM_output)
    print(model_output.shape)
    
    DeepLOB_model = Model(inputs=input_tensor, outputs= model_output)  
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1)
    
    DeepLOB_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return DeepLOB_model
```

##### Model Training

```python
DeepLOB_model = initiate_DeepLOB_model(lookback_timestep, feature_num, conv_filter_num, inception_num, LSTM_num, leaky_relu_alpha,
                          loss, optimizer, metrics)
es = EarlyStopping(monitor='val_accuracy', mode='max', patience = stop_epoch_num, verbose=1)
DeepLOB_model.fit(X_train, y_train, epochs=num_epoch, batch_size=batch_size, verbose=2, validation_data=(X_test, y_test), callbacks = [es])
```

### Results & Next Steps
Performance Metrics Comparison Table

|    | MAPE | R | Theil U |
| -- | ---  | - | ------- |
| RNN*  | 0.018  | 0.847 | 0.847|
| WSAEs-LSTM*  | 0.011  | 0.946 | 0.007
| LSTM Encoder Decoder             | 0.011  | 0.872 | 0.008 |
| LSTM Encoder Decoder + Attention | 0.0075  | 0.947 | 0.005 |
| Transformer                      | 0.0055  | 0.980 | 0.0038 |

The first two models are models from our referenced paper [A deep learning framework for financial time series using stacked autoencoders and long-short term memory](https://www.researchgate.net/publication/318991900_A_deep_learning_framework_for_financial_time_series_using_stacked_autoencoders_and_long-short_term_memory)

Daily Close Price Predictions Comparison Graph
![comparison graph](./images/comparison.jpg)

The three performance metrics and the graph match with each other and we conclude  that  among  the  three  models,  transformer model performs the best in predicting S&P500 index daily movement in our experiment setting. Although this result does not indicate transformer will always perform the best in any financial  time-series  prediction  tasks  (eg.   different  data  freqeuncy, input variables), it indicates that transformer has a high potential to be applied in financial time-series prediction tasks and should be payed more attention to in future researches on financial time series. We also plan to see whether this conclusion hold for other financial products and possibly further develop our Transformer model to generate daily adjusted investment strategy.
