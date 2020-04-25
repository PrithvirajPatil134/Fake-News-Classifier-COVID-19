## Abstract

Fake news and hoaxes have been there since before the advent of the Internet. The widely accepted definition of Internet fake news is fictitious articles deliberately fabricated to deceive readers. Nowadays, social media admins and news outlets publish fake news to increase readership or as a part of psychological warfare. With the outbreak of one of the greatest Pandemic in the history of earth, we use twitter to show our feelings, emotions and what we feel about the current situations. The media gives us live updates of how the world is dealing with the COVID-19 pandemic. What is happening at the hospitals, how the doctors, nurses and the health workers are helping to fight the disease, how scientists are working day and night in order to create a vaccination to end the pandemic.

The situation is volatile, and it is no surprise that one can easily feel frightened/ panicked in such situations. It is the human tendency to believe what we see, hear and read although most of it being just rumors created by unauthenticated and unvalidated sources. As a result of this, we get more anxious and nervous when we encounter such articles.

To address this issue, we will be designing a classifier that will accurately classify between the real news and fake news.
In our research project, we aim to analyze the news article from the kaggle datasets.  Tweets/articles extracted from twitter related to coronavirus are also used to test the models accuracy . We first extract only the required features from the dataset like title, text and label. After extracting we clean the text data by getting rid of extra whitespaces/alphanumeric characters and null values. Moreover, we stemm the words and convert all letters to lowercase alongside calculating the length of the text article. After doing so, we obliterate stopwords in the text so that the text contains only the more meaningful words that could help us classify between real and fake news. We then vectorize the words using TF-IDF and CountVectorizer retrieving at most 10,000 features after vectorizing.

As a baseline model, we first try to classify the cleaned dataset using the Naïve Bayes classifier with which we get an accuracy score of 65%. We then use Deep Learning to improve on this accuracy using the <b>Long Short Term Memory (LSTM)</b> algorithm with which we attain an accuracy of 94% after hyper-parameter tuning. We also classify using <b>H2O’s AutoML Gradient Boosting Estimator</b> to achieve an accuracy of 81%.

The results of this project will classify between real and reel news which can be very helpful to stop the spread of panic, especially during a pandemic.


## Dataset Details

We have taken 3 datasets for the model. The details of the dataset as the following:-
1. https://www.kaggle.com/c/fake-news/data
2. https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view
3. https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

All the dataset contains different columns, but they have 3 columns in common which are our prime focus. The columns are :- 
a. Text - Contains the actual text qhich we will be using to train the model and predict the fake or real news and test the accuracy
b. Title - Contains the source of the data which we will use to validate the source of the text
c. Label - The actual label of the text. Consists of values 0 and 1 indicating Real and Fake news

<b>Tweets from twitter are used for testing purpose </b>

To get the data from Twitter we are using the API from Twitter Developer account. We are ingesting 500 tweets with the Coronavirus hashtag and from a date range of 10 days.

To run the Jupyter notebook following specifications must be followed

## Specification
Hardware specs: -

1. RAM - 12 GB
2. Memory - 256 GB

For Google Colab - Google account and GPU support

NLP - installing nltk libraries - Details in the notebook<br>

TensorFlow - Keras and Tensorflow installed in the system<br>

H20 - For H20 specific Java and JDK kits must be installed along with H2O, details in the notebook<br>

Twitter Developer Account - Activated with Secret token. Details in notebook

## Cleaning the Dataset
We have read the datasets. Now we have to clean the data set. 
<br><br>
<b>
Cleaning is required because: -
</b>

1. The text data is extremely messy. 
2. Lots of noise in the data that needs to be removed
3. Many empty strings
4. NLP techniques does not recognise bad strings because when we will tokenize or vectorize it will not be performing good work on the bad data.</b>

<b>
So Cleaning of data is the first step before natural language processing. It consists of the following substeps:-
</b><br>

1. Checking missing values<br>
2. Removing records whose length is less than 40<br>
3. Removing empty texts

## Preprocessing the texts 

Data preprocessing is a data mining technique that involves transforming raw data into an understandable format. Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviors or trends, and is likely to contain many errors. Data preprocessing is a proven method of resolving such issues. Data preprocessing prepares raw data for further processing.
<br><br><b>
Pre processing steps includes the following: -</b>

1. Replace all the digits with white space
2. Converting to lower case 
3. Stemming the words in the texts
4. Taking max 10000 features and vectorizing by<br>
  a. TFIDF <br>
  b. CountVectorizer
  
<br>
We are importing <b>NLP libraries</b> for text analysis. For checking the stopwords from the text we are downloading nltk <b>STOPWORDS</b>.
For TfIdf and CountVectorizer, we are using sklearn's feature extraction library to import them. Moreover, we import <b>word_tokenize</b> from nltk library which will be used to tokenize the text data before vectorizing 

### Stemming
Much of natural language machine learning is about sentiment of the text. Stemming is a process where words are reduced to a root by removing inflection through dropping unnecessary characters, usually a suffix. There are several stemming models, including Porter and Snowball.<b> The results can be used to identify relationships and commonalities across large datasets</b>.
Using this method we will get stemmed data to feed the machine learning model through which the model can learn from root words and identify the relationships between words
<br><br>
<b>We are using Porter Stemmer to stem the text data for our dataset</b><br>
The Porter stemming algorithm (or 'Porter stemmer') is a process for removing the commoner morphological and inflexional endings from words in English. Its main use is as part of a <b>term normalisation</b> process

### TF- IDF
Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus.

TF: Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:

TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).

IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:

IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

### Naive Bayes Algorithm
Naive Bayes classification makes use of Bayes theorem to determine how probable it is that an item is a member of a category. In our use we will use it to determine if a news article belongs to the real or fake category.<br><br>
To run a Naive Bayes classifier in Scikit Learn, the categories must be numeric, so I assigned the label “1” to all fake news and the label “0” to all real news<br><br>
A Naive Bayes classifier needs to be able to calculate how many times each word appears in each document and how many times it appears in each category.Basically a matrix format where each row represents a document and each column represents a word.  Eg .[0,1,1,0,.....] [0,0,1,1….]<br><br>
To get our news articles in a matrix format, we can use Scikit Learn’s CountVectorizer. CountVectorizer creates a vector of word counts for each article to form a matrix. Each index corresponds to a word and every word appearing in the article is represented.<br><br>
We then create a dataframe of the word counts on which we train our Multinomial Naive Bayes Classifier model.<br><br>
We have used the confusion matrix as our metric to evaluate the Naive Bayes model
<br><br>
<b> Train Test Split</b><br>
Now we can split the data into train and test using Sklearn's test train split. We split the data into 75% train and 25% test

## Long Short Term Memory (LSTM)

### Overview
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points like image data, it can also process sequence of data like speech detection and Spam/ Fake News classifier.

Since Naive Bayes Classifier and Passive Aggressive Classifier did not yield the expected result, we will use LSTM to try and improve the accuracy.
As mentioned above the LSTM works on both feed forwarding and back propagation to learn from the data and can measure the sentiment of a text, so deep neural network can be a better approach to predict real or fake news and this could help in yielding better accuracy
<br>
<br>
### Environment
To use the LSTM Model we need to install Tensorflow in the system.
We can use <b>!pip install tensorflow</b> and install the tensorflow. Then we are importing different libraries like Tokenizer which is used to tokenize the text data.<b> Pad_sequence</b> is used to evenly shape the features and the labels.

### Defining the Network
The Network in keras is defined as a sequence of layer. The sequential class is the container for these layers.
The LSTM recurrent layer comprises of memory units called the LSTM. <br>

<b>Embedding Layer</b> - The Embedding layer is defined as the first hidden layer of a network. The following parameters of the embedding layers must be defined
1. input_dim: This is the size of the vocabulary in the text data. For example, if the data is integer encoded to values between 0-10, then the size of the vocabulary would be 11 words.
2. output_dim: This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word. For example, it could be 32 or 100 or even larger. Test different values for your problem.

<b>LSTM Layer</b> - This layer consists of LSTM unit. It is composed of a cell, an input gate, an output gate and a forget gate <br><br>
<b>Dropout Layer</b> - Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data. <br><br>
<b>Dense Layer</b> - A fully connected layer that often follows LSTM layers and is used for outputting a prediction is called Dense.<br>

<b>Steps to define the neural network:</b> -<br>
    
1. The first step is to create the instance of the sequence class. After defining the sequence class, we create the layer and add those to this sequence class. We are naming the class as lstm_nn_model<br>
2. The first layer is the <b>input layer</b>. In this layer we are embedding the input features and the output dimensions to form the first layer for the neural network model.<br>
3. The second layer is the <b>hidden LSTM layer</b>. The LSTM unit contains the following<br>
    a. Dropout rate is 0.3 - Drop<br>
    b. Recurrent dropout rate is 0.2<br>
    c. units training = 1<br>
4. The third layer is a <b>simple dropout layer</b> which will help in reducing oberfitting by dropping 0.4 features<br>

5. The fourth layer is the <b>Dense layer</b> of LSTM units. In this layer we are applying an activation function - <b>tanh</b><br>
6. The fifth layer is again a simple <b>dropout layer</b> with 0.5% drop out rate<br>
7. The last layer is the <b>output layer</b>. In the output layer we using another activation function <b>Softmax</b> and setting the output neurons to be 2.

### Compile the Network
After we define the neural network it should compile.<br>
<br>
This is an efficiency step. It transforms the simple layer that we define into highly complex efficient series of matrix transforms and in a format which is intended to be executed in GPU or CPU. <br>
Compilation requires a number of parameters to be specifically tailored to train the model. Specifically the optimization function and the loss function to evaluate the network which is minimized by the algorithm.<br><br>

Perhaps the most commonly used <b>optimization</b> algorithms because of their generally better performance are:

1. <b>Stochastic Gradient Descent, or ‘sgd‘</b>, that requires the tuning of a learning rate and momentum.
2. <b>ADAM, or ‘adam‘</b>, that requires the tuning of learning rate.
3. <b>RMSprop, or ‘rmsprop‘<b>, that requires the tuning of learning rate.

Here in our compilation we are using <b>"Stocastic Gradient Descend"</b> Optimizer and <b>"Sparse Categorical Cross entropy"</b> as the cost function and we are measuring the <b>accuracy</b> after each time the neural network is trained.

### Fit the Network
Once the Network is compiled it is ready to accept the weights on the training dataset. <br><br>
Fitting the network requires the training data to be specified, both a matrix of input patterns, X, and matching output patterns, y.<br><br>
The network is trained using the LSTM algorithm and optimized according to the optimization algorithm and loss function specified when compiling the model.
<br><br>
The LSTM algorithm requires that the network be trained for a specified number of epochs or exposures to the training dataset.
<br><br>
Each epoch can be partitioned into groups of input-output pattern pairs called batches. This defines the number of patterns that the network is exposed to before the weights are updated within an epoch. It is also an efficiency optimization, ensuring that not too many input patterns are loaded into memory at a time.
<br><br>
We are trainig the model to an epoch of 3. We are also provinding the validation dataset that we created to get the accuracy in the validation data.

### Evaluate the Network
Once the model is trained we can evaluate the model. <br><br>
The network is trained on training data so it do not give clear picture of the accuracy on the training dataset. So we are using the validation dataset to measure the accuracy. The validation dataset is the data that the model has never seen before.So if it performs well then we can say that the model is good.<br><br>
After training the model we get an accuracy of 57.76 % after 3 epochs in the training dataset and an accuracy of 60.4 % in the validation data.<br><br>
Clearly the accuracy is not good enough. So will train the hyper parameters to find the best parameters keeping the performance metrics as accuracy

### Hyper-Parameter Tuning
##### Tuning the network Layers

1. The first layer is kept same. As this is the input layer, we are taking all the features and the data as the input<br>
2. The Second layer -Dropout is changed to 0.2 i.ee leaving out 20% of the inputs and feeding the neural network
3. The third layer - Dropout changed to 50%. To prevent overfitting
4. Fourth layer - In this layer we changed the activation function to RELU. Relu would help in solving the problem of Vanishing Gradient.
5. Fifth layer - We are keeping this same
6. Output layer - The activation is changed to Sigmoid - Which is used for classification. Sigmoid gives 0 or 1 so it is Sigmoid would perform better that Softmax

With this we will train our model again

### Model Evaluation
After tuning the hyper parameters the accuracy of the model came to be 94% in the validation set.

<b>Key Points: -</b>
1. Dropout rate of 0.2 performed better than 0.3 in the second layer. This is because when we were using 30% it was dropping most of the features.
2. Relu activation function instead of tanh
3. In the output layer we previously used Softmax which did not perform well whereas Sigmoid generally performs well on text classification and in this model also it did
4. The optimizer we changed from SGD to Adam which helped to optimize the layers and correctly measure the cost functions

## Twitter Data

As we mentioned that we will be testing the model for the Coronavirus data scraped from twitter through developer twitter API call.

We will be focusing on the tweets which contain 'Coronavirus' hashtags and extract them using the twitter API call.<br><br>
Once this is done we create a dataframe using these tweets and then we use the LSTM model to predict the class of the tweet.
We save the prediction in a csv file.

## Auto- ML -----> H2O

H2O was the first thing that occured to our mind. “The goal of H2O is to allow simple horizontal scaling to a given problem in order to produce a solution faster.”- H2O Documentation<br><br>
We executed our models using H2O as it is a tool for rapidly turning over models, doing data munging, and building applications in a fast and scalable environment. 
<br><br>

To make h2o run in the system we need to install few dependent java packages and then do a <b>!pip install H2O


### Evaluation
We are using AUC as our metric to evaluate how well the model is performing. In this case we achieve a <b>AUC score of 81.01. 
  
## Evaluation Metric:
#### Confusion Matrix:
A confusion matrix is a summary of prediction results on a classification problem.<br> The number of correct and incorrect predictions are summarized with count values and broken down by each class.<br> This is the key to the confusion matrix. <br>The confusion matrix shows the ways in which your classification model is confused when it makes predictions.<br> It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.<br>

<b>Definition of the Terms:</b>
1. Positive (P) : Observation is positive (for example: is an apple).
2. Negative (N) : Observation is not positive (for example: is not an apple).
3. True Positive (TP) : Observation is positive, and is predicted to be positive.
4. False Negative (FN) : Observation is positive, but is predicted negative.
5. True Negative (TN) : Observation is negative, and is predicted to be negative.
6. False Positive (FP) : Observation is negative, but is predicted positive.
    Accuracy: (TP+ TN)/ (TP+TN+FP+FN)
    Precision: (TP)/(TP+FP)
    Recall: (TP)/(TP+FN)

<br>
<AUC: AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. <br>
ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s.

## Conclusion:

We have executed 4 models on the cleaned dataset that consists of tweets/articles from twitter as well.<br> <br>
<b>The 4 models are :</b>
1. Naive Bayes Classifier - 65%
2. PassiveAgresiveClassifier - 57%
3. Long Short Term Memory (LSTM) - 93%
4. H2O AutoML - 81%
		
<b>Naive Bayes Classifier</b> and <b>PassiveAggresiveClassifier </b> did not perform well on our dataset attaining an accuracy of <b>65-68%</b> on the confusion matrix even after vectorizing the words. Hence, we use <b>LSTM</b> to see how a Recurrent Neural Network model performs. After executing the model on the same dataset, we achieved an accuracy of <b>94%</b> on training data and <b>93%</b> on validation data. Hence, we run the LSTM model on twitter dataframe and store it as a csv to cross validate. LSTM model attains a good accuracy although it is quite expensive on the system. Takes up most of the system's resources to attain a good accuracy. Hence we use <b>H2O AutoML model</b> which is less expensive on the system and also helps us attain a <b>AUC of 81%</b>

