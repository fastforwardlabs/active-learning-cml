# Active Learning Workflow Demo

This is a basic dash application to demonstrate the active learning workflow. To build intuition for why active learning works, please see [this prototype](https://activelearner.fastforwardlabs.com/).

## What is Active Learning

Active learning is an iterative process, and relies on human input to build up a
smartly labeled dataset. The process typically looks like this:

* Begin with a small set of labeled data (that is all we have)
* Train a model 
* Use the trained model to run inference on a much larger pool of unlabeled data available to us
* Use a selection strategy to pick out points that are difficult for the machine
* Request labels from human
* Add these back into the labeled datapool
* Repeat the training process and iterate


## Data

We use MNIST to illustrate the workflow but only use the 10k testing dataset as
our entire dataset. We first set aside 2,000 datapoints for testing. Out of the
remaining 8,000 datapoints, we assume 1,000 are labeled while 7,000 are
unlabeled.

## How to use

There are two ways to start the application.

### As a normal python session

Just type 
```python
python app_new.py
```

### Within CDSW

This can be set up as an application.

First, specify the port in app_new.py. 

```python
app.run_server(port=os.getenv("CDSW_APP_PORT"))
```

Second, set the Subdomain in CDSW's Applications tab.

Third, enter app.py in the Script field in CDSW's Applications tab.

Fourth, start the application within CDSW.

Finally, access demo at Subdomain.ffl-4.cdsw.eng.cloudera.com

### To Do
- Reset button click should restore the graphs as well
- Why the train error > validation error
- Review that AL is adding value to the process



