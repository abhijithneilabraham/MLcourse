<html>
  <h2 align="center">Okay,now ,before starting todays class,let's revise Supervised and Unsupervised learning!(from day1.md)</h2>
  </br>
<h1 align="center">Topics</h1>
<h2 align="center">Neurons In brain</h2>
  <h4>Cells within the nervous system, called neurons, communicate with each other in unique ways. The neuron is the basic working unit of the brain, a specialized cell designed to transmit information to other nerve cells, muscle, or gland cells.
    
The brain is what it is because of the structural and functional properties of interconnected neurons. The mammalian brain contains between 100 million and 100 billion neurons, depending on the species.
</h4>
<p align="center" >
  
<img src="http://www.brainfacts.org/-/media/Brainfacts2/Brain-Anatomy-and-Function/Anatomy/Article-Images/Neuron-Illustration.jpg?la=en&hash=1D4882EC74F982F033F232C296ADF8E5EB1D9F64" label=image />
</p>
<p align="center"
<img src="http://www.quickmeme.com/img/4f/4f6426cff43d5fded163b8c294d007b022eda8a18344b21fe7be3b8c69afbc25.jpg" label=image3 </img>
</p>

<h2 align="center">Wait, are we doing machine learning or deep learning?</h2>
<p align="center">
  

<img src="https://cdn-images-1.medium.com/max/1200/1*eJfR3ui_2SsPAyMYF5R00A.jpeg"  label=image6 />
</p>
 
 
 <br> Theres a difference between machine learning and deep learning
 </br>
 <br>Machine learning is a lot of complex math and coding that, at the end of day, serves a mechanical function the same way a flashlight, a car, or a television does. When something is capable of “machine learning”, it means it’s performing a function with the data given to it, and gets progressively better at that function. It’s like if you had a flashlight that turned on whenever you said “it’s dark”, so it would recognize different phrases containing the word “dark”.</br>

<p>Deep learning
In practical terms, deep learning is just a subset of machine learning.
  Let’s go back to the flashlight example: it could be programmed to turn on when it recognizes the audible cue of someone saying the word “dark”. Eventually, it could pick up any phrase containing that word. Now if the flashlight had a deep learning model, it could maybe figure out that it should turn on with the cues “I can’t see” or “the light switch won’t work”. A deep learning model is able to learn through its own method of computing – its own “brain”, if you will. </p>
 <h4 align="center"> So,any clue as to why I explained about a neuron earlier?</h4>
 
 
<h2 align="center">Neural Networks and perceptrons</h2>
<h4>Perceptron is a single layer neural network and a multi-layer perceptron is called Neural Networks.</h4>
Perceptron is a linear classifier (binary). Also, it is used in supervised learning
<br>But how does it work?</br>
<br>The perceptron works on these simple steps</br>


<br>a. All the inputs x are multiplied with their weights w. Let’s call it k.
<img src="https://cdn-images-1.medium.com/max/800/1*_Zy1C83cnmYUdETCeQrOgA.png" label=image4 align="center" />
</br>
b. Add all the multiplied values and call them Weighted Sum.

<img src="https://cdn-images-1.medium.com/max/800/1*xFd9VQnUM1H0kiCENsoYxg.gif" align="center" label=image5 />
<br>c. Apply that weighted sum to the correct Activation Function.
</br>
For Example : Unit Step Activation Function.
<img src="https://cdn-images-1.medium.com/max/800/1*0iOzeMS3s-3LTU9hYH9ryg.png"  label=image6 align="center" />

<h2 align='center'> Neural Network Layout </h2>
<img align='center' label=image6 src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/280px-Colored_neural_network.svg.png" />
<br> In a neural network architecure, there is an input layer, some hidden layers and an output Layer. </br>
<br>The weighted outputs from input layer is passed to the input of next layer, and so on, and we will get the predicted output in numerical form at the output layer. </br>


<h2>Overfitting</h2>
<br>Overfitting is when the trained model memorizes the undesirable patterns or noise from the training data-set. This is due to too much training or learning capacity(too many hidden layers or neurons in each layer). The consequence of overfitting is that, model cannot generalize to samples outside its training set, which overall reduces the performance of the model. To determine whether the model is overfitting, during the training, compare the loss value on the training and testing set. If the loss on the test set is much larger than in the training set, then the model is overfitting, specially if the training loss is low. However, it is also normal that the test loss is slightly larger than training loss.</br>


<img src="https://cdn-images-1.medium.com/max/1200/1*cdvfzvpkJkUudDEryFtCnA.png" label=image2 />
<h2>Qualities of good features</h2>
<br><h3>let's read about feature engineering</h3>
</br>

[Click here to read about Feature Engineering](https://developers.google.com/machine-learning/crash-course/representation/feature-engineering)

<br>Suppose, we are given a data “flight date time vs status”. Then, given the date-time data, we have to predict the status of the flight.
<img src="https://cdn-images-1.medium.com/max/800/1*4uxZB7gAd-1Vm4nVwtouwg.png" align=center label=image7 />
</br>
As the status of the flight depends on the hour of the day, not on the date-time. We will create the new feature “Hour_Of_Day”. Using the “Hour_Of_Day” feature, the machine will learn better as this feature is directly related to the status of the flight.
<img src="https://cdn-images-1.medium.com/max/800/1*U5ZAUIb_9nq2EqlhIC1WfA.png" align=center label=image8 />
Here, creating the new feature “Hour_Of_Day” is the feature engineering.
<h2 align="center"> Feature crosses</h2>

[Read this article about Feature crossing](https://developers.google.com/machine-learning/crash-course/feature-crosses/encoding-nonlinearity)
<h2 align="center">One Hot Encoding</h2>

[Read this awesome explanation about One Hot Encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)

<h2 align="center">Regularisation</h2>
<img src="https://cdn-images-1.medium.com/max/800/1*zYfwoRcih4jzyDP3j3aVmQ.png" label=image3 />
<h4> What is the need for regularisation? </h4>

This is a form of regression, that constrains/ regularizes or shrinks the coefficient estimates towards zero. In other words, this technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting.


<h2> Activation Functions</h2>
To model a nonlinear problem, we can directly introduce a nonlinearity. We can pipe each hidden layer node through a nonlinear function.
<img src="https://missinglink.ai/wp-content/uploads/2018/11/activationfunction-1.png" />
In the model represented by the following graph, the value of each node in Hidden Layer 1 is transformed by a nonlinear function before being passed on to the weighted sums of the next layer. This nonlinear function is called the activation function

[click to read more about activation functions](https://developers.google.com/machine-learning/crash-course/introduction-to-neural-networks/anatomy)

<h3> Here are some activation functions</h3>
<ol>
  <li>ReLu -Rectified linear units</li>
  <li>Sigmoid or Logistic</li>
  <li>Tanh — Hyperbolic tangent</li>
</ol>




<h2>Forward and Backward Propogation</h2>
<p align="center" >
<img src="https://www.bogotobogo.com/python/scikit-learn/images/NeuralNetwork2-Forward-Propagation/NN-with-components-w11-etc.png" label=lalala />
</p>

In neural networks, you forward propagate to get the output and compare it with the real value to get the error.

Now, to minimize the error, you propagate backwards by finding the derivative of error with respect to each weight and then subtracting this value from the weight value.

The basic learning that has to be done in neural networks is training neurons when to get activated. Each neuron should activate only for particular type of inputs and not all inputs. Therefore, by propagating forward you see how well your neural network is behaving and find the error. After you find out that your network has error, you back propagate and use a form of gradient descent to update new values of weights. Then, you will again forward propagate to see how well those weights are performing and then will backward propagate to update the weights. This will go on until you reach some minima for error value.
<h2>Training</h2>
<br>What happens when you try to learn something for an examination? You dont exactly memorise everything as such, right?
You create a model inside your head, so that when a question is asked , you can answer accordingly using the model you have inside your head.</br>
<br>The training data must contain the correct answer, which is known as a target or target attribute. The learning algorithm finds patterns in the training data that map the input data attributes to the target (the answer that you want to predict), and it outputs an ML model that captures these patterns.

You can use the ML model to get predictions on new data for which you do not know the target. For example, let's say that you want to train an ML model to predict if an email is spam or not spam. You would provide the network with training data that contains emails for which you know the target (that is, a label that tells whether an email is spam or not spam). The network would train an ML model by using this data, resulting in a model that attempts to predict whether new email will be spam or not spam.
</br>


</html>
