## Active Learner

Supervised machine learning, while powerful, needs labeled data to be
effective. Active learning reduces the number of labeled examples needed to
train a model, saving time and money while obtaining comparable performance to
models trained with much more data.


Active learning is an iterative process, and relies on human input to build up a
smartly labeled dataset. The process typically looks like this:

* Begin with a small set of labeled data (that is all we have)
* Train a model 
* Use the trained model to run inference on a much larger pool of unlabeled data available to us
* Use a selection strategy to pick out points that are difficult for the machine
* Request labels from human
* Add these back into the labeled datapool
* Repeat the training process and iterate


In this demo, we use the MNIST dataset to illustrate the Active Learning
workflow. To start, we train a convolutional network using only 1000 labeled
datapoints. You can kick off training in Tab "Training". Once the model has
completed training, we can visualize the embeddings of the convolutional network
by using UMAP to project the high dimensional representation down to 2D. 

In the "Training" tab, you can specify a selection strategy to surface
datapoints that are difficult for the machine. The selected datapoints are
overlaid on the 2D visualization as yellow dots. In the "Labeling" tab, you can
provide a label for these datapoints.

Once the labels are submitted, they are added back to the original 1000 labeled
datapoints (bringing the total labeled datapoints to be 1010). The model is now
ready to be trained again.

This application is meant to serve as a complement to the
[prototype](https://activelearner.fastforwardlabs.com/) for the report
we released on Learning with Limited Labeled Data.