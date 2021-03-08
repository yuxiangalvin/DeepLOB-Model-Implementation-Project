
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

There has been previous researches which focus on limit orderbook data. However most of them applied static feature extract method that mainly based on domain expertise and conventions. This static methods include Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA), etc. However, the DeepLOB model applies a dynamic feature ectraction approach through applying Deep Learning architecture (CNN + Inception Module). It also uses an additional LSTM to capture additional dependence that is not captured by the CNN + Inception Module.



Firstly this DeepLOB model takes limit orderbook data as inputs, specifically, at each time point, it takes in a limit orderbook snippet - the lowest 10 ask levels and the highest 10 bid levels. Each level has one price data and one size data. 

### Model Details

Here are brief description of our three models:

For the baseline, we used a LSTM based model. It's a standard seq2seq architecture, in which two recurrent neural networks work together to transform one sequence to another. The encoder condenses an input sequence into a hidden vector, and the decoder unfolds that vector into a new sequence. In our settings, the encoder is a 3-layer LSTM whose inputs are 30 days of stock data, the decoder is a single LSTM cell which will be used repeatedly to generate the stock index of the future 5 days. The optimizer is an Adam optimizer with default parameters, we used MSE loss to train the model.

Our LSTM + Attention model basically has the same architecture as the LSTM model, but it has one more attention layer which can calculate the weight using the encoder output. In our model we used Bahdanau Attention. The other settings are the same as the first LSTM model, a defualt Adam optimzier and MSE criterion.

Our transformer model is build based on the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). We replace the embedding layer of NLP tasks with a 1-D convolutional layer for projecting the time-series input into a length=dmodel vector.   Our transformermodel uses 6 layers transformer encoder and 3 layers decoder,8 heads self-attention, and dmodel=512. We use an SGD optimizer with CosineAnnealing learning rate decay and MSEloss to train the model.

### Data Used

This graph shows the candidate variables we explored for our model.

![Input Variables](./images/variables_used.PNG)

We used daily S&P500 data between 2008/07/02 and 2016/09/30. This dataset is from our referenced paper [A deep learning framework for financial time series using stacked autoencoders and long-short term memory](https://www.researchgate.net/publication/318991900_A_deep_learning_framework_for_financial_time_series_using_stacked_autoencoders_and_long-short_term_memory)

### Expriment  Method

All 3 models are constructed to tackle the same task. They take 30 days of input varaible data and predict the close price daily percentage move of the next day. We keep the task same to compare the performance of three model studctures. We separated the data into three sets - 80% for training set, 10% for validation set and 10% for testing set. training set and validation set are generated by randomly split, and test set contains the data from specific date. 

We applied subsection prediction method \cite{subsection_method}. This method has three parts. Firstly, we used training set to train the model. Secondly, we used the validation set to verify the optimal model setting. Lastly, we used testing set to measure the performance of the model. We used 3 classical performance metrics for financial time series data - MAPE, R and Theil U


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
