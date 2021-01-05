# Active Learning Workflow Demo

This is a basic Dash application to demonstrate the active learning workflow. 
To build intuition for why active learning works, please see [this prototype](https://activelearner.fastforwardlabs.com/).

## What is Active Learning

Active learning is an iterative process, and relies on human input to build up a
smartly labeled dataset. The process typically looks like this:

* Begin with a small set of labeled data (that is all we have)
* Train a model 
* Use the trained model to run inference on a much larger pool of unlabeled data 
available to us
* Use a selection strategy to pick out points that are difficult for the machine
* Request labels from human
* Add these back into the labeled datapool
* Repeat the training process and iterate


## Data

We use MNIST to illustrate the workflow but only use the 10k testing dataset as
our entire dataset. We first set aside 2,000 datapoints for validation/testing. 
Out of the remaining 8,000 datapoints, we allow user to select 100, 500 or 1,000 as 
initial labeled examples while the rest are unlabeled. The user can then provide 
labels for the 10 shortlisted examples (based on the selection strategy) from the 
remaining training examples and continue to train a model with the additional data 
points. In the long run the model performance should differ based on the selection strategy 
employed.

## How to use

There are two ways to start the application.

### As a normal python session

Just type 
  ```python
  python app_new.py
  ```

### Within CDSW

This can be set up as an application.

- First, specify the port in app_new.py. 

  ```python
  app.run_server(port=os.getenv("CDSW_APP_PORT"))
  ```

- Second, set the Subdomain in CDSW's Applications tab.

- Third, enter app.py in the Script field in CDSW's Applications tab.

- Fourth, start the application within CDSW.

- Finally, access demo at `subdomain.ffl-4.cdsw.eng.cloudera.com`

### To Do
- Why the train error > validation error - done, fixed
- Add refresh graphs button instead of flickering? - fixed
- What's wrong with the epoch updates now? - fixed
- Reset button click should restore the graphs as well - fixed
- Review that AL is adding value to the process - done
  Yes, it does. For instance, say we have 100 examples and train a model for 60 epochs
  with random strategy, the val accuracy is then 78%. If we use entropy strategy we 
  get around 83% by the end of 70 epochs. Although close this should ideally get better 
  with more epochs .
- The "label this image" option should disappear after training, either point to a different image
  or just have "click a data point ..." message shown
- Exception/ bug: - fixed
  ```
    reset_clicks:  0
    Exception on /_dash-update-component [POST]
    Traceback (most recent call last):
      File "/home/cdsw/.local/lib/python3.6/site-packages/flask/app.py", line 2447, in wsgi_app
        response = self.full_dispatch_request()
      File "/home/cdsw/.local/lib/python3.6/site-packages/flask/app.py", line 1952, in full_dispatch_request
        rv = self.handle_user_exception(e)
      File "/home/cdsw/.local/lib/python3.6/site-packages/flask/app.py", line 1821, in handle_user_exception
        reraise(exc_type, exc_value, tb)
      File "/home/cdsw/.local/lib/python3.6/site-packages/flask/_compat.py", line 39, in reraise
        raise value
      File "/home/cdsw/.local/lib/python3.6/site-packages/flask/app.py", line 1950, in full_dispatch_request
        rv = self.dispatch_request()
      File "/home/cdsw/.local/lib/python3.6/site-packages/flask/app.py", line 1936, in dispatch_request
        return self.view_functions[rule.endpoint](**req.view_args)
      File "/home/cdsw/.local/lib/python3.6/site-packages/dash/dash.py", line 1050, in dispatch
        response.set_data(func(*args, outputs_list=outputs_list))
      File "/home/cdsw/.local/lib/python3.6/site-packages/dash/dash.py", line 995, in add_context
        _validate.validate_multi_return(output_spec, output_value, callback_id)
      File "/home/cdsw/.local/lib/python3.6/site-packages/dash/_validate.py", line 116, in validate_multi_return
        callback_id, repr(output_value)
    dash.exceptions.InvalidCallbackReturnValue: The callback ..train.n_clicks.. is a multi-output.
    Expected the output type to be a list or tuple but got:
    None.
  ```
- Handle index point exception for "orig_x", it complained about size being 100 - fixed


