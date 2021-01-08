Supervised machine learning, while powerful, needs labeled data to be
effective. Active learning reduces the number of labeled examples needed to
train a model, saving time and money while obtaining comparable performance to
models trained with much more data.

In this demo, we use the MNIST dataset to illustrate the Active Learning
workflow. To start, we train a convolutional neural network using only 1000 labeled
datapoints. You can kick off training in Tab "Training". Once the model has
completed training, we can visualize the embeddings of the convolutional neural network
by using UMAP to project the high dimensional representation down to 2D. 

This application is meant to serve as a complement to the
[prototype](https://activelearner.fastforwardlabs.com/) for the report
we released on Learning with Limited Labeled Data.