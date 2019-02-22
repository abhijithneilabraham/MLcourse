# Welcome!
<p align="center">
  <img src="https://github.com/abhijithneilabraham/MLcourse/blob/master/ML1.png"
       height="500" width="500"alt="Keralaai tf study jam"/>
</p>




1. Intro to machine learning
2. Regression Vs Classification
3. Linear Regression
4. Reducing Loss
5. Training and Testing sets
6. Logistic Regression
7. First Steps with some frameworks
8. Tasks

## Intro to machine learning
Do you wonder how life works? How humans make and adapt and learn? We cannot learn anything unless we have a data as input,which are gained mostly through our sense organs.
Well, do give machines some data, give them the power of wisdom, and ,let them learn things their own way!
Almost like, watching a child grow right infront of you. :)
https://thispersondoesnotexist.com

### Types of machine learning algorithms:
- Supervised Learning

<p>
  <img src="https://github.com/GopikrishnanSasikumar/Intro-to-Machine-Learning-Workshop/blob/master/images-2.png"alt="supervised learning"/>
</p>
Applications:
</br> Information extraction
</br>Object recognition in image processing
</br>Pattern recognition
</br>Speech recognition


- Unsupervised Learning(I kind of like to imagine it as a human named tarzan who lives in jungle seeing two different animals!)
<h3>
  As you can see,
  the most common method for unsupervised learning is clustering.
<p>
  <img src="https://github.com/GopikrishnanSasikumar/Intro-to-Machine-Learning-Workshop/blob/master/unsupervised_learning.png" height="200" width="400" alt="Un-Supervised Learning"/>
</p>
 </br>The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data.

- Reinforcement Learning(action,state and rewards)

<p>
  <img src="https://github.com/GopikrishnanSasikumar/Intro-to-Machine-Learning-Workshop/blob/master/1*HvoLc50Dpq1ESKuejhICHg.png" alt="Reinforcement Learning"/>
</p>
</br>Continuos learning process,with self improvement,based on rewards.
</br> Application Example:Self Driving car.


In supervised learning we create “models”, a model is basically a function that takes in simple inputs and produces useful predictions. Here we have features and labels in the dataset.

### Features:

A feature is an input variable—the x variable .
. A simple machine learning project might use a single feature, while a more sophisticated machine learning project could use millions of features.The learning is done by understanding the features.

Features of sem exam scores for predicting ML model can be,

- Total number of hours spent studying the day before exam.
- previous knowledge from +2 topics.(Brilliant pala students please step back!!)
- Love for PUBG :wink:

### Labels:

A label is the thing we're predicting. It can be the price of a product, class probability in a classification.

### Examples:
  An example is a particular instance of data, x. (We put x in boldface to indicate that it is a vector.) We break examples into two categories:

</br>	&#8226;labeled examples
</br>	&#8226;unlabeled examples
</br>A labeled example includes both feature(s) and the label. That is:
```
 labeled examples: {features, label}: (x, y)
```
</br>Use labeled examples to train our Model



### Regression Vs Classification
</br>Supervised learning problems can be further grouped into regression and classification problems.

<h3>Classification:</h3> </br> A classification problem is when the output variable is a category, such as “red” or “blue” or “disease” and “no disease”.
<h3>Regression:</h3> </br> A regression problem is when the output variable is a real value, such as “dollars” or “weight”.

A regression model is used to predict continuous values.

For example,

- The probability of winning the chicken dinner after several retries.(PUBG fans can relate!)
- Price of petrol in India.

A classification model predicts discrete values. It can make predictions that answer questions like,

- Is this an image of a cat or dog or Priya prakash warrier.
- Predicting whether a movie belongs to DC or Marvel(based on the dark scree(and humour) may be). 



<h2>Okay guys,time to check your understanding!</h2>
<h4>Click on the link below</h4>
https://developers.google.com/machine-learning/crash-course/framing/check-your-understanding

### Linear Regression

Linear regression is a method for finding the straight line or hyperplane that best fits a set of points.

The line equation is,

```
y = mx + b 

```

In machine learning we use this convention instead,

```
y' = b + w1x1
```
Where,

- y' is the label we are predicting.
- b is the bias.(does the same thing what a constant c does to the line equation.Determines if to pass through origin or not)
- w1 is the weight of feature 1. Weight is the same concept as the "slope" 
 in the traditional equation of a line.
- x1 is a feature (a known input).

To predict, just substitute the x1 values to the trained model.

A sophisticated model can use more than one features.

```
y' = b + w1x1 + w2x2 + w3x3 + .... + wNxN
here, x1,x2,x3 are the different different features which predict for the label.
```
<h4> Lets look at a sample XY plot.</h4>
<img src="https://nowenlightenme.files.wordpress.com/2018/09/regression-best-fit-on-new_x-and-new_y.png?w=640" label=image9 />


1. **Training** a model simply means learning (determining) good values for all the weights and the bias from labeled examples. 

2. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called **empirical risk minimization**.

3. Loss is the penalty for a bad prediction. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater. 

4. The goal of training a model is to find a set of weights and biases that have low loss, on average, across all examples.

First we have to find the loss.
<h2>Cost Function</h2>

**L2 Loss/square loss** is a popular loss function. It is the given as

```
= the square of the difference between the label and the prediction
= (observation - prediction(x))2
= (y - y')2
```
We need a function that will minimize the parameters over our dataset. One common function that is often used is mean squared error, which measure the difference between the estimator (the dataset) and the estimated value (the prediction). 
**Mean square error (MSE)** is the average squared loss per example over the whole dataset.

<p align="center">
  <img src="https://github.com/GopikrishnanSasikumar/Intro-to-Machine-Learning-Workshop/blob/master/mse.png"alt="MSE"/>
</p>

## Reducing Loss
Reducing the loss is similar to the **"Hot and cold game"** kids play!(Hot if youre nearer,cold if youre not.)

A Machine Learning model is trained by starting with an initial guess for the weights and bias and iteratively adjusting those guesses until learning the weights and bias with the lowest possible loss.

### Gradient Descent
Gradient Descent is a general function for minimizing a function, in this case the Mean Squared Error cost function.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/0*rBQI7uBhBKE8KT-X.png" height="300" width="500" alt="Gradient Descent"/>
</p>
To explain it very briefly, it is a way to determine how well the machine learning model has performed given the different values of each parameters.
Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model. Parameters refer to coefficients in Linear Regression and weights in neural networks.
In the above figure, we must reach the steepest point,i.e, the bottom most point here,which is the local minimum. We have to converge to that value, and then we can say, with minimum loss, we can predict the output.


</br>
</br>
<p align="center">
<img src="https://cdn-images-1.medium.com/max/1200/1*8wpL_9SdB1rRuN8ll-u5hA.png" label=image4 />
</p>
### Learning Rate

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/0*QwE8M4MupSdqA3M4.png" height="300" width="500" alt="Gradient Descent"/>
</p>
We take steps in gradient descent to converge to the local minimum. But learning rate also shouldnt be to high or too low, which result in values other than local minimum.



## Training and Testing Sets
<br>

<p align="center">
  <img src="http://blogs-images.forbes.com/janakirammsv/files/2017/04/ML-FaaS-1.png" height="360" width="740" alt="train-test divison"/>
</p>

1. Our goal is to create a machine learning model that generalizes well to new data. 

2. We train the model using a Training set and the test set act as a proxy for new data!


<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/1*-8_kogvwmL1H6ooN1A1tsQ.png" height="360" width="740" alt="train-test divison"/>
</p>

- **training set** — a subset to train a model.
- **test set** — a subset to test the trained model.

## Logistic Regression

1. Many problems require a probability estimate as output.
2. Logistic regression is an extremely efficient mechanism for calculating probabilities.

For example, consider that the probability of coconut falling on someone's head while walking through a field is 0.05.
Then over the year 18 accidents will happen in that field because of coconut!
```
P(thenga|day) = 0.05
coconut falling on head =
0.05*365 
~= 18
```


3. a sigmoid function, defined as follows, produces output that always falls between 0 and 1.
So,the aim of logistic regression is to get a probabilistic value between 0 and 1, and you can see that in the graph 

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSw89aMI5qlmImjji48z1agmJOhIJDSJvZgrHD9WPR4q783tMEMkw" height="100" width="150" alt="sigmoid function"/>
</p>

Where,

```
y = w1x1 + w2x2 + ... wNxN
```
and p is the predicted output.



<p align="center">
  <img src="https://developers.google.com/machine-learning/crash-course/images/SigmoidFunction.png" height="300" width="600" alt="sigmoid graph"/>
</p>



### Loss function for Logistic regression is Log Loss

<p align="center">
  <img src="https://github.com/GopikrishnanSasikumar/Intro-to-Machine-Learning-Workshop/blob/master/logloss.png" alt="sigmoid graph"/>
</p>


<h2>Comparison to linear regression</h2>
Given data on time spent studying and exam scores. Linear Regression and logistic regression can predict different things:

Linear Regression could help us predict the student’s test score on a scale of 0 - 100. Linear regression predictions are continuous (numbers in a range).
Logistic Regression could help use predict whether the student passed or failed. Logistic regression predictions are discrete (only specific values or categories are allowed). We can also view probability scores underlying the model’s classifications.
<h2>Decision Boundary</h2>



## First Steps with Tensorflow

Tensorflow is a computational framework for building machine learning models. TensorFlow provides a variety of different toolkits that allow you to construct models at your preferred level of abstraction. You can use lower-level APIs to build models by defining a series of mathematical operations. Alternatively, you can use higher-level APIs (like tf.estimator) to specify predefined architectures, such as linear regressors or neural networks.

Tensorflow consist of,

- A graph protocol buffer

- A runtime that executes the distributed graph

### Tensorflow hierarchy

|                                 |                                       |
|---------------------------------|---------------------------------------|
| Estimator (tf.estimator)        | High-level, OOP API.                  |  
| tf.layers/tf.losses/tf.metrics  | Libraries for common model components.|
| TensorFlow	                    | Lower-level APIs                      |

<h3>There will always be awesome deep learning frameworks for your use,like, keras,pytorch,tensorflow,etc,but
  I think we should start with something very light and understandable with the theory we have studied until now.</h2>



<h2>Why don't you try with scikit learn framework with me?</h2>
<h1> Scikit learn</h1>
</br>Simple and efficient tools for data mining and data analysis
</br>Accessible to everybody, and reusable in various contexts
</br>Built on NumPy, SciPy, and matplotlib
</br>Open source, commercially usable - BSD license

</br> Let's now check a code and visualize and train a small custom dataset to understand how linear regression works.
</br>https://github.com/abhijithneilabraham/Scikit-matplotlib-pandas/blob/master/linear_model.py


<p>## Tasks
</br>1)https://scikit-learn.org/stable/tutorial/basic/tutorial.html . try this and other example tasks with scikit learn with built-in datasets.

</br>2)Hey! what if you changed the dataset, i.e,change the training and test set values and parameters so that you can use it in your own application? May the best idea win!
</p>


