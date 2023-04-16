# Parametric RELU ANNs

## Previously I have talked about ANNs in vectorized form. In this project , I would like to explore how to create ANNs with Parametric RELU as the activation for the hidden layers.

<aside>
⚠️ Do refer to this link for my first project on creating ANNs using vectorization approach : https://leeyz888.github.io/6-hidden-layer/

</aside>

### Alternatively , can use this link to access the HTML version of this md document : ##

## 1)Dataset description

### First , let’s talk a bit about the dataset I plan to use.

- It’s the MNST dataset I used previously.
- The difference is that for this dataset the images have dimensions 28x28, i.e each data has 784 features. The images are in grayscale color.
- It will be loaded via using the Keras library. The dataset has 60000 training data and 10000 testing data .
- The data will NOT be used as numpy arrays but will be loaded into Python as a Tensor object.

## 2)Network architecture

- Do refer to my first project to get a general idea on how I plan to structure my network.
- In general , the activation for the hidden layers will now be Parametric ReLU, instead of just normal ReLU. The activation for the output however will still be SoftMax.
- The parametric ReLU has the formula $max(X,0)+((alpha)*(min(X,0))$ .
- I.e , for positive values , the Parametric ReLU will output the value unchanged.
- For negative values however , the Parametric ReLU will output the negative value multiplied by a parameter alpha, denoted by α.

## 3) Derivation of the derivative of the loss w.r.t to Parametric ReLU

- Assuming that the derivative of the loss function w.r.t to each activation in the hidden layer of the ANN exists.
- There are 2 types of derivations that can be used, depending on whether the alpha is shared across all hidden nodes, or that for each hidden node there exists a separate alpha.
- The former is known as **shared-alpha**, while the latter is known as **channel wise alpha**.

### The parametric ReLU in piecewise form, along with its derivative.

![截屏 2023-04-16 下午1.13.26.png](Parametric%20RELU%20ANNs%20842025a496a148c0a47ef6ba46f03cd6/%25E6%2588%25AA%25E5%25B1%258F_2023-04-16_%25E4%25B8%258B%25E5%258D%25881.13.26.png)

<aside>
⚠️ ${Z_i}$ is the output of the hidden layer i before it is passed into the activation , which in this case is the Parametric ReLU.

</aside>

## Let’s start by deriving the formula for shared-alpha.

$$
\begin{equation} \frac{\partial L}{\partial \alpha_{h_i}} = \sum_{j=1}^{H_{i+1}}\sum_{k=1}^{H_{i}} \sum_{m=1}^{M} \frac{\partial L}{\partial z_{h_{i+1}j}^{(m)}} \frac{\partial z_{h_{i+1}j}^{(m)}}{\partial a_{h_{i}k}^{(m)}} \frac{\partial a_{h_{i}k}^{(m)}}{\partial \alpha_{{h_i}}} \end{equation}
$$

$$
\begin{equation} \frac{\partial L}{\partial \alpha_{h_i}} = \sum_{j=1}^{H_{i+1}}\sum_{k=1}^{H_{i}} \sum_{m=1}^{M} {dz}^{m}_{h_{i+1},{j}} W_{{h_{i},k}\to {h_{i+1},j}} \frac{\partial a_{h_{i}k}^{(m)}}{\partial \alpha_{{h_i}}} \end{equation}
$$

<aside>
⚠️ This looks very similar to the derivative of the loss w.r.t the hidden layer before the activation function, i.e $\frac{\partial L}{\partial z_{{h_i},m}}$. In other words, the derivative of the loss w.r.t the activation is very useful as it can then be differentiated w.r.t the parameters in the function , making them learnable.

</aside>

<aside>
⚠️ $\frac{\partial a_{h_{i}k}^{(m)}}{\partial \alpha_{{h_i},k}}$ =0 for positive $Z_{h_{i+1},m}$ , and is equal to $Z_{h_{i+1},m}$ for non positive $Z_{h_{i+1},m}$.

</aside>

## In vectorized form, the formula is

![截屏 2023-04-16 下午1.44.48.png](Parametric%20RELU%20ANNs%20842025a496a148c0a47ef6ba46f03cd6/%25E6%2588%25AA%25E5%25B1%258F_2023-04-16_%25E4%25B8%258B%25E5%258D%25881.44.48.png)

## For channel wise alpha:

$$
\begin{equation} \frac{\partial L}{\partial \alpha_{h_i,k}} = \sum_{j=1}^{H_{i+1}}\ \sum_{m=1}^{M} \frac{\partial L}{\partial z_{h_{i+1}j}^{(m)}} \frac{\partial z_{h_{i+1}j}^{(m)}}{\partial a_{h_{i}k}^{(m)}} \frac{\partial a_{h_{i}k}^{(m)}}{\partial \alpha_{{h_i},k}} \end{equation}
$$

$$
\begin{equation} \frac{\partial L}{\partial \alpha_{h_i,k}} = \sum_{j=1}^{H_{i+1}} \sum_{m=1}^{M} {dz}^{m}_{h_{i+1},{j}} W_{{h_{i},k}\to {h_{i+1},j}} \frac{\partial a_{h_{i}k}^{(m)}}{\partial \alpha_{{h_i},k}} \end{equation}
$$

## In vectorized form , the formula is:

![IMG_0364.jpg](Parametric%20RELU%20ANNs%20842025a496a148c0a47ef6ba46f03cd6/IMG_0364.jpg)

## 4) Implementation in TensorFlow by constructing a 6 hidden layer ANN

### Shared alpha:

```jsx
import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import softmax as sm
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from math import sqrt
from math import log

class NeuralNet(tf.keras.Model):
    def __init__(self, num_features, num_hidden1,num_hidden2,num_hidden3,num_hidden4,num_hidden5,num_hidden6, alpha, alpha1,alpha2,alpha3,alpha4,alpha5,alpha6 ,max_epochs, num_output, _EPSILON):

      super(NeuralNet, self).__init__()
        
      self.num_features = num_features
      self.num_hidden1 = num_hidden1
      self.num_hidden2 = num_hidden2
      self.num_hidden3 = num_hidden3
      self.num_hidden4 = num_hidden4
      self.num_hidden5 = num_hidden5
      self.num_hidden6 = num_hidden6
      self.alpha = alpha
      self.alpha1 = tf.Variable(tf.constant(alpha1),name='alpha1')
      self.alpha2 = tf.Variable(tf.constant(alpha2),name='alpha2')
      self.alpha3 = tf.Variable(tf.constant(alpha3),name='alpha3')
      self.alpha4 = tf.Variable(tf.constant(alpha4),name='alpha4')
      self.alpha5 = tf.Variable(tf.constant(alpha5),name='alpha5')
      self.alpha6 = tf.Variable(tf.constant(alpha6),name='alpha6')
      self.max_epochs = max_epochs
      self.num_output = num_output
      self._EPSILON = _EPSILON
      self.loss = []
      self.trainingaccur = []
      self.devaccur = []
        
      self.Weights_Input_to_H1 = tf.Variable(tf.random.normal([self.num_hidden1, self.num_features], mean=0.0, stddev=0.1),name='Weights_Input_to_H1')
      self.Bias_Input_to_H1 = tf.Variable(tf.zeros([self.num_hidden1, 1]),name='Bias_Input_to_H1')
      self.Weights_H1_to_H2 = tf.Variable(tf.random.normal([self.num_hidden2, self.num_hidden1], mean=0.0, stddev=0.1),name='Weights_H1_to_H2')
      self.Bias_H1_to_H2 = tf.Variable(tf.zeros([self.num_hidden2, 1]),name='Bias_H1_to_H2')
      self.Weights_H2_to_H3 = tf.Variable(tf.random.normal([self.num_hidden3, self.num_hidden2], mean=0.0, stddev=0.1),name='Weights_H2_to_H3')
      self.Bias_H2_to_H3 = tf.Variable(tf.zeros([self.num_hidden3, 1]),name='Bias_H2_to_H3')
      self.Weights_H3_to_H4 = tf.Variable(tf.random.normal([self.num_hidden4, self.num_hidden3], mean=0.0, stddev=0.1),name='Weights_H3_to_H4')
      self.Bias_H3_to_H4 = tf.Variable(tf.zeros([self.num_hidden4, 1]),name='Bias_H3_to_H4')
      self.Weights_H4_to_H5 = tf.Variable(tf.random.normal([self.num_hidden5, self.num_hidden4], mean=0.0, stddev=0.1),name='Weights_H4_to_H5')
      self.Bias_H4_to_H5 = tf.Variable(tf.zeros([self.num_hidden5, 1]),name='Bias_H4_to_H5')
      self.Weights_H5_to_H6 = tf.Variable(tf.random.normal([self.num_hidden6, self.num_hidden5], mean=0.0, stddev=0.1),name='Weights_H5_to_H6')
      self.Bias_H5_to_H6 = tf.Variable(tf.zeros([self.num_hidden6, 1]),name='Bias_H5_to_H6')
      self.Weights_H6_to_output = tf.Variable(tf.random.normal([self.num_output, self.num_hidden6], mean=0.0, stddev=0.1),name='Weights_H6_to_output')
      self.Bias_H6_to_output = tf.Variable(tf.zeros([self.num_output, 1]),name='Bias_H6_to_output')
        
      self.dWeights_Input_to_H1 = tf.Variable(tf.zeros([self.num_hidden1, self.num_features]),name='dWeights_Input_to_H1')
      self.dBias_Input_to_H1 = tf.Variable(tf.zeros([self.num_hidden1, 1]),name='dBias_Input_to_H1')
      self.dWeights_H1_to_H2 = tf.Variable(tf.zeros([self.num_hidden2, self.num_hidden1]),name='dWeights_H1_to_H2')
      self.dBias_H1_to_H2 = tf.Variable(tf.zeros([self.num_hidden2, 1]),name='dBias_H1_to_H2')
      self.dWeights_H2_to_H3 = tf.Variable(tf.zeros([self.num_hidden3, self.num_hidden2]),name='dWeights_H2_to_H3')
      self.dBias_H2_to_H3 = tf.Variable(tf.zeros([self.num_hidden3, 1]),name='dBias_H2_to_H3')
      self.dWeights_H3_to_H4 = tf.Variable(tf.zeros([self.num_hidden4, self.num_hidden3]),name='dWeights_H3_to_H4')
      self.dBias_H3_to_H4 = tf.Variable(tf.zeros([self.num_hidden4, 1]),name='dBias_H3_to_H4')
      self.dWeights_H4_to_H5 = tf.Variable(tf.zeros([self.num_hidden5, self.num_hidden4]),name='dWeights_H4_to_H5')
      self.dBias_H4_to_H5 = tf.Variable(tf.zeros([self.num_hidden5, 1]),name='dBias_H4_to_H5')
      self.dWeights_H5_to_H6 = tf.Variable(tf.zeros([self.num_hidden6, self.num_hidden5]),name='dWeights_H5_to_H6')
      self.dBias_H5_to_H6 = tf.Variable(tf.zeros([self.num_hidden6, 1]),name='dBias_H5_to_H6')
      self.dWeights_H6_to_output = tf.Variable(tf.zeros([self.num_output, self.num_hidden6]),name='dWeights_H6_to_output')
      self.dBias_H6_to_output = tf.Variable(tf.zeros([self.num_output, 1]),name='dBias_H6_to_output')
      
    def relU(self, X):
        return tf.where(X<=0, 0, X)

    def Para_relU(self, alpha, X):
        return tf.where(X<=0, alpha*X, X)

    def Para_deriv_wrt_X(self, alpha, X):
        return tf.where(X<=0, alpha, 1)

    def Para_deriv_wrt_alpha(self, alpha, X):
        return tf.where(X<=0, X, 0)

    def deriv(self, X):
        return tf.where(X<=0, 0, 1)

    def softmax(self, x):
        e = x - tf.reduce_max(x, axis=0)
        return tf.exp(e) / tf.reduce_sum(tf.exp(e), axis=0)

    def forward(self, X):
        self.z1 = tf.matmul(self.Weights_Input_to_H1, X) + self.Bias_Input_to_H1
        self.a1 = self.Para_relU(self.alpha1, self.z1) 
        self.z2 = tf.matmul(self.Weights_H1_to_H2, self.a1) + self.Bias_H1_to_H2
        self.a2 = self.Para_relU(self.alpha2, self.z2) 
        self.z3 = tf.matmul(self.Weights_H2_to_H3, self.a2) + self.Bias_H2_to_H3
        self.a3 = self.Para_relU(self.alpha3, self.z3) 
        self.z4 = tf.matmul(self.Weights_H3_to_H4, self.a3) + self.Bias_H3_to_H4
        self.a4 = self.Para_relU(self.alpha4, self.z4)
        self.z5 = tf.matmul(self.Weights_H4_to_H5, self.a4) + self.Bias_H4_to_H5
        self.a5 = self.Para_relU(self.alpha5, self.z5)
        self.z6 = tf.matmul(self.Weights_H5_to_H6, self.a5) + self.Bias_H5_to_H6
        self.a6 = self.Para_relU(self.alpha6, self.z6)
        self.z7 = tf.matmul(self.Weights_H6_to_output, self.a6) + self.Bias_H6_to_output
        self.a7 = tf.nn.softmax(self.z7, axis=0)
        return self.a7

       

    def backprop(self, X, t):
      
        self.dz7=(tf.reshape(self.a7, [self.num_output,-1])-tf.reshape(t, [self.num_output,-1]))/((X.shape[1]))
        self.dBias_H6_to_output=tf.reduce_sum(self.dz7, axis=1, keepdims=True)
        self.dWeights_H6_to_output=tf.matmul(self.dz7, tf.transpose(self.a6))
        self.dz6=(tf.matmul(tf.transpose(self.Weights_H6_to_output), self.dz7)) * (self.Para_deriv_wrt_X(self.alpha6,self.z6))
        self.dalpha6=(tf.matmul(tf.transpose(self.Weights_H6_to_output), self.dz7)) * (self.Para_deriv_wrt_alpha(self.alpha6,self.z6))
        self.dalpha6_scalar=tf.reduce_sum(self.dalpha6)
        self.dBias_H5_to_H6=tf.reduce_sum(self.dz6, axis=1, keepdims=True)
        self.dWeights_H5_to_H6=tf.matmul(self.dz6, tf.transpose(self.a5))
        self.dz5=(tf.matmul(tf.transpose(self.Weights_H5_to_H6), self.dz6)) * (self.Para_deriv_wrt_X(self.alpha5,self.z5))
        self.dalpha5=(tf.matmul(tf.transpose(self.Weights_H5_to_H6), self.dz6)) * (self.Para_deriv_wrt_alpha(self.alpha5,self.z5))
        self.dalpha5_scalar=tf.reduce_sum(self.dalpha5)
        self.dBias_H4_to_H5=tf.reduce_sum(self.dz5, axis=1, keepdims=True)
        self.dWeights_H4_to_H5=tf.matmul(self.dz5, tf.transpose(self.a4))
        self.dz4=(tf.matmul(tf.transpose(self.Weights_H4_to_H5), self.dz5)) * (self.Para_deriv_wrt_X(self.alpha4,self.z4))
        self.dalpha4=(tf.matmul(tf.transpose(self.Weights_H4_to_H5), self.dz5)) * (self.Para_deriv_wrt_alpha(self.alpha4,self.z4))
        self.dalpha4_scalar=tf.reduce_sum(self.dalpha4)
        self.dBias_H3_to_H4=tf.reduce_sum(self.dz4, axis=1, keepdims=True)
        self.dWeights_H3_to_H4=tf.matmul(self.dz4, tf.transpose(self.a3))
        self.dz3=(tf.matmul(tf.transpose(self.Weights_H3_to_H4), self.dz4)) * (self.Para_deriv_wrt_X(self.alpha3,self.z3))
        self.dalpha3=(tf.matmul(tf.transpose(self.Weights_H3_to_H4), self.dz4)) * (self.Para_deriv_wrt_alpha(self.alpha3,self.z3))
        self.dalpha3_scalar=tf.reduce_sum(self.dalpha3)
        self.dBias_H2_to_H3=tf.reduce_sum(self.dz3, axis=1, keepdims=True)
        self.dWeights_H2_to_H3=tf.matmul(self.dz3, tf.transpose(self.a2))
        self.dz2=(tf.matmul(tf.transpose(self.Weights_H2_to_H3), self.dz3)) * (self.Para_deriv_wrt_X(self.alpha2,self.z2))
        self.dalpha2=(tf.matmul(tf.transpose(self.Weights_H2_to_H3), self.dz3)) * (self.Para_deriv_wrt_alpha(self.alpha2,self.z2))
        self.dalpha2_scalar=tf.reduce_sum(self.dalpha2)
        self.dBias_H1_to_H2=tf.reduce_sum(self.dz2, axis=1, keepdims=True)
        self.dWeights_H1_to_H2=tf.matmul(self.dz2, tf.transpose(self.a1))
        self.dz1=(tf.matmul(tf.transpose(self.Weights_H1_to_H2), self.dz2)) * (self.Para_deriv_wrt_X(self.alpha1,self.z1))
        self.dalpha1=(tf.matmul(tf.transpose(self.Weights_H1_to_H2), self.dz2)) * (self.Para_deriv_wrt_alpha(self.alpha1,self.z1))
        self.dalpha1_scalar=tf.reduce_sum(self.dalpha1)
        self.dBias_Input_to_H1=tf.reduce_sum(self.dz1, axis=1, keepdims=True)
        self.dWeights_Input_to_H1=tf.matmul(self.dz1, tf.transpose(X))

    def fit(self, x_train_data, y_train_data,x_dev_data,y_dev_data):
       
        
        
        for step in range(self.max_epochs):
      
            self.forward(x_train_data)
            self.backprop(x_train_data, y_train_data)
            self.Bias_H1_to_H2.assign_sub(self.alpha * self.dBias_H1_to_H2)
            self.Weights_H1_to_H2.assign_sub(self.alpha * self.dWeights_H1_to_H2)
            self.Bias_H2_to_H3.assign_sub(self.alpha * self.dBias_H2_to_H3)
            self.Weights_H2_to_H3.assign_sub(self.alpha * self.dWeights_H2_to_H3)
            self.Bias_H3_to_H4.assign_sub(self.alpha * self.dBias_H3_to_H4)
            self.Weights_H3_to_H4.assign_sub(self.alpha * self.dWeights_H3_to_H4)
            self.Bias_H4_to_H5.assign_sub(self.alpha * self.dBias_H4_to_H5)
            self.Weights_H4_to_H5.assign_sub(self.alpha * self.dWeights_H4_to_H5)
            self.Bias_H5_to_H6.assign_sub(self.alpha * self.dBias_H5_to_H6)
            self.Weights_H5_to_H6.assign_sub(self.alpha * self.dWeights_H5_to_H6)
            self.Bias_H6_to_output.assign_sub(self.alpha * self.dBias_H6_to_output)
            self.Weights_H6_to_output.assign_sub(self.alpha * self.dWeights_H6_to_output)
            self.Bias_Input_to_H1.assign_sub(self.alpha * self.dBias_Input_to_H1)
            self.Weights_Input_to_H1.assign_sub(self.alpha * self.dWeights_Input_to_H1)
            self.alpha1.assign_sub(self.alpha * self.dalpha1_scalar)
            self.alpha2.assign_sub(self.alpha * self.dalpha2_scalar)
            self.alpha3.assign_sub(self.alpha * self.dalpha3_scalar)
            self.alpha4.assign_sub(self.alpha * self.dalpha4_scalar)
            self.alpha5.assign_sub(self.alpha * self.dalpha5_scalar)
            self.alpha6.assign_sub(self.alpha * self.dalpha6_scalar)
            print(step)

            if step % 100 == 0:
              self.CCloss = tf.keras.losses.categorical_crossentropy(tf.transpose(y_train_data), tf.transpose(self.forward(x_train_data)), from_logits=False, label_smoothing=0)
              self.CCloss=tf.reduce_mean(self.CCloss).numpy()
              self.trainingaccuracy = accuracy_score(np.argmax(y_train_data.numpy(),axis=0), np.argmax(self.forward(x_train_data).numpy(),axis=0))
              self.devaccuracy = accuracy_score(np.argmax(y_dev_data.numpy(),axis=0), np.argmax(self.forward(x_dev_data).numpy(),axis=0))
              print(f'step: {step},  loss: {self.CCloss}') 
              print(self.trainingaccuracy)
              print(self.devaccuracy)
              self.loss.append(self.CCloss)
              self.trainingaccur.append(self.trainingaccuracy)
              self.devaccur.append(self.devaccuracy)
              print(tf.reduce_mean(((self.dz7*60000)**2)))
              print(self.alpha1.numpy(),self.dalpha1_scalar.numpy())
              print(self.alpha2.numpy(),self.dalpha2_scalar.numpy())
              print(self.alpha3.numpy(),self.dalpha3_scalar.numpy())
              print(self.alpha4.numpy(),self.dalpha4_scalar.numpy())
              print(self.alpha5.numpy(),self.dalpha5_scalar.numpy())
              print(self.alpha6.numpy(),self.dalpha6_scalar.numpy())
```

### Channel-wise alpha

```jsx
import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import softmax as sm
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from math import sqrt
from math import log

class NeuralNet(tf.keras.Model):
    def __init__(self, num_features, num_hidden1,num_hidden2,num_hidden3,num_hidden4,num_hidden5,num_hidden6, alpha, alpha1,alpha2,alpha3,alpha4,alpha5,alpha6 ,max_epochs, num_output, _EPSILON):

      super(NeuralNet, self).__init__()
        
      self.num_features = num_features
      self.num_hidden1 = num_hidden1
      self.num_hidden2 = num_hidden2
      self.num_hidden3 = num_hidden3
      self.num_hidden4 = num_hidden4
      self.num_hidden5 = num_hidden5
      self.num_hidden6 = num_hidden6
      self.alpha = alpha
      self.alpha1 = tf.Variable(alpha1 * tf.ones([self.num_hidden1, 1]))
      self.alpha2 = tf.Variable(alpha2 * tf.ones([self.num_hidden2, 1]))
      self.alpha3 = tf.Variable(alpha3 * tf.ones([self.num_hidden3, 1]))
      self.alpha4 = tf.Variable(alpha4 * tf.ones([self.num_hidden4, 1]))
      self.alpha5 = tf.Variable(alpha5 * tf.ones([self.num_hidden5, 1]))
      self.alpha6 = tf.Variable(alpha6 * tf.ones([self.num_hidden6, 1]))
      self.max_epochs = max_epochs
      self.num_output = num_output
      self._EPSILON = _EPSILON
      self.loss = []
      self.trainingaccur = []
      self.devaccur = []
        
      self.Weights_Input_to_H1 = tf.Variable(tf.random.normal([self.num_hidden1, self.num_features], mean=0.0, stddev=0.1))
      self.Bias_Input_to_H1 = tf.Variable(tf.zeros([self.num_hidden1, 1]))
      self.Weights_H1_to_H2 = tf.Variable(tf.random.normal([self.num_hidden2, self.num_hidden1], mean=0.0, stddev=0.1))
      self.Bias_H1_to_H2 = tf.Variable(tf.zeros([self.num_hidden2, 1]))
      self.Weights_H2_to_H3 = tf.Variable(tf.random.normal([self.num_hidden3, self.num_hidden2], mean=0.0, stddev=0.1))
      self.Bias_H2_to_H3 = tf.Variable(tf.zeros([self.num_hidden3, 1]))
      self.Weights_H3_to_H4 = tf.Variable(tf.random.normal([self.num_hidden4, self.num_hidden3], mean=0.0, stddev=0.1))
      self.Bias_H3_to_H4 = tf.Variable(tf.zeros([self.num_hidden4, 1]))
      self.Weights_H4_to_H5 = tf.Variable(tf.random.normal([self.num_hidden5, self.num_hidden4], mean=0.0, stddev=0.1))
      self.Bias_H4_to_H5 = tf.Variable(tf.zeros([self.num_hidden5, 1]))
      self.Weights_H5_to_H6 = tf.Variable(tf.random.normal([self.num_hidden6, self.num_hidden5], mean=0.0, stddev=0.1))
      self.Bias_H5_to_H6 = tf.Variable(tf.zeros([self.num_hidden6, 1]))
      self.Weights_H6_to_output = tf.Variable(tf.random.normal([self.num_output, self.num_hidden6], mean=0.0, stddev=0.1))
      self.Bias_H6_to_output = tf.Variable(tf.zeros([self.num_output, 1]))
        
      self.dWeights_Input_to_H1 = tf.Variable(tf.zeros([self.num_hidden1, self.num_features]))
      self.dBias_Input_to_H1 = tf.Variable(tf.zeros([self.num_hidden1, 1]))
      self.dWeights_H1_to_H2 = tf.Variable(tf.zeros([self.num_hidden2, self.num_hidden1]))
      self.dBias_H1_to_H2 = tf.Variable(tf.zeros([self.num_hidden2, 1]))
      self.dWeights_H2_to_H3 = tf.Variable(tf.zeros([self.num_hidden3, self.num_hidden2]))
      self.dBias_H2_to_H3 = tf.Variable(tf.zeros([self.num_hidden3, 1]))
      self.dWeights_H3_to_H4 = tf.Variable(tf.zeros([self.num_hidden4, self.num_hidden3]))
      self.dBias_H3_to_H4 = tf.Variable(tf.zeros([self.num_hidden4, 1]))
      self.dWeights_H4_to_H5 = tf.Variable(tf.zeros([self.num_hidden5, self.num_hidden4]))
      self.dBias_H4_to_H5 = tf.Variable(tf.zeros([self.num_hidden5, 1]))
      self.dWeights_H5_to_H6 = tf.Variable(tf.zeros([self.num_hidden6, self.num_hidden5]))
      self.dBias_H5_to_H6 = tf.Variable(tf.zeros([self.num_hidden6, 1]))
      self.dWeights_H6_to_output = tf.Variable(tf.zeros([self.num_output, self.num_hidden6]))
      self.dBias_H6_to_output = tf.Variable(tf.zeros([self.num_output, 1]))
      
    def relU(self, X):
        return tf.where(X<=0, 0, X)

    def Para_relU(self, alpha, X):
        return tf.where(X<=0, alpha*X, X)

    def Para_deriv_wrt_X(self, alpha, X):
        return tf.where(X<=0, alpha, 1)

    def Para_deriv_wrt_alpha(self, alpha, X):
        return tf.where(X<=0, X, 0)

    def deriv(self, X):
        return tf.where(X<=0, 0, 1)

    def softmax(self, x):
        e = x - tf.reduce_max(x, axis=0)
        return tf.exp(e) / tf.reduce_sum(tf.exp(e), axis=0)

    def forward(self, X):
        self.z1 = tf.matmul(self.Weights_Input_to_H1, X) + self.Bias_Input_to_H1
        self.a1 = self.Para_relU(self.alpha1, self.z1) 
        self.z2 = tf.matmul(self.Weights_H1_to_H2, self.a1) + self.Bias_H1_to_H2
        self.a2 = self.Para_relU(self.alpha2, self.z2) 
        self.z3 = tf.matmul(self.Weights_H2_to_H3, self.a2) + self.Bias_H2_to_H3
        self.a3 = self.Para_relU(self.alpha3, self.z3) 
        self.z4 = tf.matmul(self.Weights_H3_to_H4, self.a3) + self.Bias_H3_to_H4
        self.a4 = self.Para_relU(self.alpha4, self.z4)
        self.z5 = tf.matmul(self.Weights_H4_to_H5, self.a4) + self.Bias_H4_to_H5
        self.a5 = self.Para_relU(self.alpha5, self.z5)
        self.z6 = tf.matmul(self.Weights_H5_to_H6, self.a5) + self.Bias_H5_to_H6
        self.a6 = self.Para_relU(self.alpha6, self.z6)
        self.z7 = tf.matmul(self.Weights_H6_to_output, self.a6) + self.Bias_H6_to_output
        self.a7 = tf.nn.softmax(self.z7, axis=0)
        return self.a7

       

    def backprop(self, X, t):
      
        self.dz7=(tf.reshape(self.a7, [self.num_output,-1])-tf.reshape(t, [self.num_output,-1]))/((X.shape[1]))
        self.dBias_H6_to_output=tf.reduce_sum(self.dz7, axis=1, keepdims=True)
        self.dWeights_H6_to_output=tf.matmul(self.dz7, tf.transpose(self.a6))
        self.dz6=(tf.matmul(tf.transpose(self.Weights_H6_to_output), self.dz7)) * (self.Para_deriv_wrt_X(self.alpha6,self.z6))
        self.dalpha6=(tf.matmul(tf.transpose(self.Weights_H6_to_output), self.dz7)) * (self.Para_deriv_wrt_alpha(self.alpha6,self.z6))
        self.dalpha6_scalar=tf.reduce_sum(self.dalpha6, axis=1, keepdims=True)
        self.dBias_H5_to_H6=tf.reduce_sum(self.dz6, axis=1, keepdims=True)
        self.dWeights_H5_to_H6=tf.matmul(self.dz6, tf.transpose(self.a5))
        self.dz5=(tf.matmul(tf.transpose(self.Weights_H5_to_H6), self.dz6)) * (self.Para_deriv_wrt_X(self.alpha5,self.z5))
        self.dalpha5=(tf.matmul(tf.transpose(self.Weights_H5_to_H6), self.dz6)) * (self.Para_deriv_wrt_alpha(self.alpha5,self.z5))
        self.dalpha5_scalar=tf.reduce_sum(self.dalpha5, axis=1, keepdims=True)
        self.dBias_H4_to_H5=tf.reduce_sum(self.dz5, axis=1, keepdims=True)
        self.dWeights_H4_to_H5=tf.matmul(self.dz5, tf.transpose(self.a4))
        self.dz4=(tf.matmul(tf.transpose(self.Weights_H4_to_H5), self.dz5)) * (self.Para_deriv_wrt_X(self.alpha4,self.z4))
        self.dalpha4=(tf.matmul(tf.transpose(self.Weights_H4_to_H5), self.dz5)) * (self.Para_deriv_wrt_alpha(self.alpha4,self.z4))
        self.dalpha4_scalar=tf.reduce_sum(self.dalpha4, axis=1, keepdims=True)
        self.dBias_H3_to_H4=tf.reduce_sum(self.dz4, axis=1, keepdims=True)
        self.dWeights_H3_to_H4=tf.matmul(self.dz4, tf.transpose(self.a3))
        self.dz3=(tf.matmul(tf.transpose(self.Weights_H3_to_H4), self.dz4)) * (self.Para_deriv_wrt_X(self.alpha3,self.z3))
        self.dalpha3=(tf.matmul(tf.transpose(self.Weights_H3_to_H4), self.dz4)) * (self.Para_deriv_wrt_alpha(self.alpha3,self.z3))
        self.dalpha3_scalar=tf.reduce_sum(self.dalpha3, axis=1, keepdims=True)
        self.dBias_H2_to_H3=tf.reduce_sum(self.dz3, axis=1, keepdims=True)
        self.dWeights_H2_to_H3=tf.matmul(self.dz3, tf.transpose(self.a2))
        self.dz2=(tf.matmul(tf.transpose(self.Weights_H2_to_H3), self.dz3)) * (self.Para_deriv_wrt_X(self.alpha2,self.z2))
        self.dalpha2=(tf.matmul(tf.transpose(self.Weights_H2_to_H3), self.dz3)) * (self.Para_deriv_wrt_alpha(self.alpha2,self.z2))
        self.dalpha2_scalar=tf.reduce_sum(self.dalpha2, axis=1, keepdims=True)
        self.dBias_H1_to_H2=tf.reduce_sum(self.dz2, axis=1, keepdims=True)
        self.dWeights_H1_to_H2=tf.matmul(self.dz2, tf.transpose(self.a1))
        self.dz1=(tf.matmul(tf.transpose(self.Weights_H1_to_H2), self.dz2)) * (self.Para_deriv_wrt_X(self.alpha1,self.z1))
        self.dalpha1=(tf.matmul(tf.transpose(self.Weights_H1_to_H2), self.dz2)) * (self.Para_deriv_wrt_alpha(self.alpha1,self.z1))
        self.dalpha1_scalar=tf.reduce_sum(self.dalpha1, axis=1, keepdims=True)
        self.dBias_Input_to_H1=tf.reduce_sum(self.dz1, axis=1, keepdims=True)
        self.dWeights_Input_to_H1=tf.matmul(self.dz1, tf.transpose(X))

    def fit(self, x_train_data, y_train_data,x_dev_data,y_dev_data):
       
        
        
        for step in range(self.max_epochs):
      
            self.forward(x_train_data)
            self.backprop(x_train_data, y_train_data)
            self.Bias_H1_to_H2.assign_sub(self.alpha * self.dBias_H1_to_H2)
            self.Weights_H1_to_H2.assign_sub(self.alpha * self.dWeights_H1_to_H2)
            self.Bias_H2_to_H3.assign_sub(self.alpha * self.dBias_H2_to_H3)
            self.Weights_H2_to_H3.assign_sub(self.alpha * self.dWeights_H2_to_H3)
            self.Bias_H3_to_H4.assign_sub(self.alpha * self.dBias_H3_to_H4)
            self.Weights_H3_to_H4.assign_sub(self.alpha * self.dWeights_H3_to_H4)
            self.Bias_H4_to_H5.assign_sub(self.alpha * self.dBias_H4_to_H5)
            self.Weights_H4_to_H5.assign_sub(self.alpha * self.dWeights_H4_to_H5)
            self.Bias_H5_to_H6.assign_sub(self.alpha * self.dBias_H5_to_H6)
            self.Weights_H5_to_H6.assign_sub(self.alpha * self.dWeights_H5_to_H6)
            self.Bias_H6_to_output.assign_sub(self.alpha * self.dBias_H6_to_output)
            self.Weights_H6_to_output.assign_sub(self.alpha * self.dWeights_H6_to_output)
            self.Bias_Input_to_H1.assign_sub(self.alpha * self.dBias_Input_to_H1)
            self.Weights_Input_to_H1.assign_sub(self.alpha * self.dWeights_Input_to_H1)
            self.alpha1.assign_sub(self.alpha * self.dalpha1_scalar)
            self.alpha2.assign_sub(self.alpha * self.dalpha2_scalar)
            self.alpha3.assign_sub(self.alpha * self.dalpha3_scalar)
            self.alpha4.assign_sub(self.alpha * self.dalpha4_scalar)
            self.alpha5.assign_sub(self.alpha * self.dalpha5_scalar)
            self.alpha6.assign_sub(self.alpha * self.dalpha6_scalar)
            print(step)

            if step % 100 == 0:
              #self.CCloss = tf.keras.losses.categorical_crossentropy(tf.transpose(y_train_data), tf.transpose(self.forward(x_train_data)), from_logits=False, label_smoothing=0)
              self.CCloss=log_loss(np.transpose(y_train_data.numpy()),np.transpose(self.forward(x_train_data).numpy()),eps=self._EPSILON,normalize=True)
              self.trainingaccuracy = accuracy_score(np.argmax(y_train_data.numpy(),axis=0), np.argmax(self.forward(x_train_data).numpy(),axis=0))
              self.devaccuracy = accuracy_score(np.argmax(y_dev_data.numpy(),axis=0), np.argmax(self.forward(x_dev_data).numpy(),axis=0))
              print(f'step: {step},  loss: {self.CCloss}') 
              print(self.trainingaccuracy)
              print(self.devaccuracy)
              self.loss.append(self.CCloss)
              self.trainingaccur.append(self.trainingaccuracy)
              self.devaccur.append(self.devaccuracy)
              print(tf.reduce_mean(((self.dz7*60000)**2)).numpy())
```

## 5) Miscellaneous codes

```python
import numpy as np
import tensorflow as tf
from keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

```python
X_train=tf.reshape(x_train, [60000, 784])
X_test=tf.reshape(x_test, [10000, 784])
```

```python
X_train_floating=tf.cast(X_train, dtype=tf.float32)
X_test_floating=tf.cast(X_test, dtype=tf.float32)
```

```python
X_train_floating_scaled=((X_train_floating))/(255.0)
X_test_floating_scaled=((X_test_floating))/(255.0)
```

```python
y_train_encoded=tf.one_hot(y_train,10)
y_test_encoded=tf.one_hot(y_test,10)
```

```python
X_train_floating_scaled=tf.transpose(X_train_floating_scaled)
X_test_floating_scaled=tf.transpose(X_test_floating_scaled)
```

```python
y_train_encoded=tf.transpose(y_train_encoded)
y_test_encoded=tf.transpose(y_test_encoded)
```

<aside>
⚠️ Run these to load an preprocess the MNST dataset.

</aside>

```python
numHidden1 = 400 # number of hidden nodes
numHidden2 = 400# number of hidden nodes
numHidden3 = 400# number of hidden nodes
numHidden4 = 400# number of hidden nodes
numHidden5 = 400# number of hidden nodes
numHidden6 = 400# number of hidden nodes
num_features = X_train_floating_scaled.shape[0]
num_output = y_train_encoded.shape[0]
max_epochs = 10000
alpha = 0.001
epsilon=0.00000000001
alpha1=0.001
alpha2=0.001
alpha3=0.001
alpha4=0.001
alpha5=0.001
alpha6=0.001
NN = NeuralNet(num_features, numHidden1,numHidden2,numHidden3,numHidden4,numHidden5,numHidden6, alpha, alpha1,alpha2,alpha3,alpha4,alpha5,alpha6 ,max_epochs, num_output, epsilon)
```

<aside>
⚠️ Run this to initialize the class.

</aside>

```python
NN.fit(X_train_floating_scaled,y_train_encoded,X_test_floating_scaled,y_test_encoded)
```

## 6) Analysis of the performances of both ANNs

## Let’s investigate the losses , training and dev accuracies of both networks.

### For channel-wise alpha:

```python
import matplotlib.pyplot as plt
x_loss=range(0,len(NN.loss)*100,100)

line1=plt.plot(x_loss,NN.loss,linestyle='-',label='training loss')  

plt.title('Training loss for channel wise alpha')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
legend = plt.legend(loc='best', shadow=True)
```

![Untitled](Parametric%20RELU%20ANNs%20842025a496a148c0a47ef6ba46f03cd6/Untitled.png)

```python
x_training_accur=range(0,len(NN.trainingaccur)*100,100)
x_devaccur=range(0,len(NN.devaccur)*100,100)

line1=plt.plot(x_training_accur,NN.trainingaccur,linestyle='-',label='training accuracy') 
line2=plt.plot(x_devaccur,NN.devaccur,linestyle='-',label='dev accuracy')
                              
plt.title('Training and Dev Accuracies for channel wise alpha')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='best', shadow=True)
```

![Untitled](Parametric%20RELU%20ANNs%20842025a496a148c0a47ef6ba46f03cd6/Untitled%201.png)

### Shared alpha

```python
import matplotlib.pyplot as plt
x_loss=range(0,len(NN.loss)*100,100)

line1=plt.plot(x_loss,NN.loss,linestyle='-',label='training loss')  

plt.title('Training loss for shared alpha')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
legend = plt.legend(loc='best', shadow=True)
```

![Untitled](Parametric%20RELU%20ANNs%20842025a496a148c0a47ef6ba46f03cd6/Untitled%202.png)

```python
x_training_accur=range(0,len(NN.trainingaccur)*100,100)
x_devaccur=range(0,len(NN.devaccur)*100,100)

line1=plt.plot(x_training_accur,NN.trainingaccur,linestyle='-',label='training accuracy') 
line2=plt.plot(x_devaccur,NN.devaccur,linestyle='-',label='dev accuracy')
                              
plt.title('Training and Dev Accuracies for shared alpha')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='best', shadow=True)
```

![Untitled](Parametric%20RELU%20ANNs%20842025a496a148c0a47ef6ba46f03cd6/Untitled%203.png)

# 6) Conclusion

<aside>
⚠️ We managed to build ANNs with parametric ReLU as the activation for the hidden layers. The main takeaway from this project is that the derivative of the loss w.r.t the activation can be extremely useful as we can then differentiate it w.r.t the parameters in the activation, therefore making them learnable parameters.

</aside>
