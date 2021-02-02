Supervised machine learning, while powerful, needs labeled data to be
effective. Active learning reduces the number of labeled examples needed to
train a model, saving time and money while obtaining comparable performance to
models trained with much more data.   

In this application, we use the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/) 
to illustrate the active learning workflow. The workflow begins by training a 
convolutional neural network to recognize what digit is present in each image 
using only a small sample of labeled examples. We then compute the loss and 
accuracy of this model on a hold out validation set and are presented with ten 
unseen, ambiguous (as defined by the model) examples which we can manually 
assign labels for. Those ten examples are then added into the original training 
set which helps the model learn to distinguish between ambiguous classes more 
effectively. This process can be repeated until an acceptable level of performance 
is achieved. For step by step instructions, click the "Learn More" button below.
    
This application is meant to serve as a complement to the [prototype](https://activelearner.fastforwardlabs.com/) 
for the report we released on Learning with Limited Labeled Data. To build an 
intuition for why active learning works, please see our blogpost - 
[A guide to learning with limited labeled data](https://blog.cloudera.com/a-guide-to-learning-with-limited-labeled-data/) 