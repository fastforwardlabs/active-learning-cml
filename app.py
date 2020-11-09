import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from PIL import Image, ImageEnhance
from dataset import get_dataset, get_handler
from io import BytesIO
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from torchvision import transforms
from model import get_net
from data import Data
from train import Train
from sample import Sample
import base64
import torch
import umap
import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#server = flask.Flask('app')
#server.secret_key = os.environ.get('secret_key', 'secret')

"""
Set general arguements
"""

dataset_name = 'MNIST'
data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,),
                                                          (0.3081,))])
handler = get_handler(dataset_name)
n_classes = 10
n_epoch = 10

"""
Define model architecture
"""
net = get_net(dataset_name)

"""
Get dataset, set data handlers
Use 10k training dataset from MNIST as the entire dataset
Further split into Test/Train, then split train into labeled and unlabeled
X (Training): 1000 datapoints
X_NOLB (unlabeled datapool): 7000
X_TE (Testing): 2000 datapoints
X_TOLB: Datapoints that require human labeling
"""
# FIX, do this in data object?
_, _, X_INIT, Y_INIT = get_dataset(dataset_name)

X_TR, X_TE, Y_TR, Y_TE = train_test_split(X_INIT, Y_INIT,
                                          test_size=0.2,
                                          random_state=42,
                                          shuffle=True)

X_NOLB, X, Y_NOLB, Y = train_test_split(X_TR, Y_TR,
                                        test_size=0.125,
                                        random_state=42,
                                        shuffle=True)

X_TOLB = torch.empty([10, 28, 28], dtype=torch.uint8)

"""
Make data and train objects
"""

data = Data(X, Y, X_TE, Y_TE, X_NOLB, X_TOLB,
            data_transform, handler, n_classes)

"""
VARs to control embedding figures
"""
EMB_HISTORY = None
"""
Set random seeds for UMAP
Initialize UMAP reducer
"""
np.random.seed(42)
reducer = umap.UMAP(random_state=42)

# intro text
intro_text = """

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


"""

training_text = """
* First, click "Train" button to start model training
* When training is completed, a 2D visualization of the embeddings will be rendered
* The datapoints that require human labeling are overlaid on the 2D visualization as yellow dots
* Next, provide labels in Tab "Labeling" and click "Submit"
* Train the model again

"""


def reset_data():
    global data
    data = Data(X, Y, X_TE, Y_TE, X_NOLB, X_TOLB,
                data_transform, handler, n_classes)
    return data


def train_model():
    train_obj = Train(net, handler, n_epoch, data)
    train_obj.train()
    return train_obj


def data_to_label(strategy):
    print('getting new data to label')
    sample = Sample(train_obj.clf, data)
    if strategy == "random":
        print('random')
        X_TOLB, X_NOLB = sample.random(10)
    elif strategy == "entropy":
        print('entropy')
        X_TOLB, X_NOLB = sample.entropy(10)
    elif strategy == "entropydrop":
        print('entropy dropout')
        X_TOLB, X_NOLB = sample.entropy_dropout(10, 5)
    return (X_TOLB, X_NOLB)


def numpy_to_b64(array, scalar=True):
    # Convert from 0-1 to 0-255
    if scalar:
        array = np.uint8(255 * array)

    array[np.where(array == 0)] = 255

    im_pil = Image.fromarray(array)

    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return "data:image/png;base64," + im_b64


def create_img(arr, shape=(28, 28)):
    arr = arr.reshape(shape).astype(np.float64)
    image_b64 = numpy_to_b64(arr)
    return image_b64


app.layout = html.Div(id='main-content', className='app-body', children=[
    dcc.Loading(className='fig-loading', children=html.Div(
        id='main-graph',
        children=dcc.Graph(id='embedding',
                           style={'bgcolor': 'rgba(0,0,0,0)'},
                           config={'scrollZoom': True},)
        )),
    html.Div(id="store-nclicks", style={'display': 'none'}),
    # html.Div(id="store-nclicks"),
    # html.Div(id="store-strategy"),   
    #html.Div(id="store-strategy", style={'display': 'none'}),    
    # html.Div(id="train-obj", style={'display': 'none'}),
    # html.Div(id="store-nclicks"),
    html.Div([
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(label='About',
                    value='tab-0',
                    children=html.Div(className='control-tab', children=[
                        html.H4(className='tab-0', children='The Active Learning Workflow'),
                        dcc.Markdown(intro_text),
                        ])),
            dcc.Tab(label='Training', value='tab-1'),
            dcc.Tab(label='Labeling', value='tab-2'),
        ]),
        html.Div(id='tabs-content'),
    ])
])


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-2':
        return html.Div(children=[
            html.H2(
                children='Difficult Datapoints',
            ),
            html.H4(
                children='Please label the handwritten images below (0-9)'
                
            ),

            html.Img(
                id='image0',
                className='image',
                src=create_img(data.X_TOLB[0].numpy())),
            
            html.Label('Label'),
            dcc.Input(id="label0", value=0, type='number'),
            
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Img(
                id='image1',
                className='image',
                src=create_img(data.X_TOLB[1].numpy())),
            
            html.Label('Label'),
            dcc.Input(id="label1", value=0, type='number'),
            
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            
            html.Img(
                id='image2',
                className='image',
                src=create_img(data.X_TOLB[2].numpy())),
            
            html.Label('Label'),
            dcc.Input(id="label2", value=0, type='number'),
            
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            
            html.Img(
                id='image3',
                className='image',
                src=create_img(data.X_TOLB[3].numpy())),
            
            html.Label('Label'),
            dcc.Input(id="label3", value=0, type='number'),
            
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            
            html.Img(
                id='image4',
                className='image',
                src=create_img(data.X_TOLB[4].numpy())),
            
            html.Label('Label'),
            dcc.Input(id="label4", value=0, type='number'),
            
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            
            html.Img(
                id='image5',
                className='image',
                src=create_img(data.X_TOLB[5].numpy())),
            
            html.Label('Label'),
            dcc.Input(id="label5", value=0, type='number'),
            
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            
            html.Img(
                id='image6',
                className='image',
                src=create_img(data.X_TOLB[6].numpy())),
            
            html.Label('Label'),
            dcc.Input(id="label6", value=0, type='number'),
            
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            
            html.Img(
                id='image7',
                className='image',
                src=create_img(data.X_TOLB[7].numpy())),
            
            html.Label('Label'),
            dcc.Input(id="label7", value=0, type='number'),
            
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            
            html.Img(
                id='image8',
                className='image',
                src=create_img(data.X_TOLB[8].numpy())),
            
            html.Label('Label'),
            dcc.Input(id="label8", value=0, type='number'),
            
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
    
            
            html.Img(
                id='image9',
                className='image',
                src=create_img(data.X_TOLB[9].numpy())),
            
            html.Label('Label'),
            dcc.Input(id="label9", value=0, type='number'),
            html.Br(),
            html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
            html.Div(id="number-output")
            
        ])
    elif tab == 'tab-1':
        return html.Div([
            html.H4('Train Models'),
            dcc.Markdown(training_text),
            html.Label('Selection Strategy'),
            dcc.RadioItems(
                id='al-strategy',
                options=[
                    {'label': 'Random', 'value': 'random'},
                    {'label': 'Entropy', 'value': 'entropy'},
                    {'label': 'Entropy with Dropout', 'value': 'entropydrop'}
                ],
                value='random'
            ),
            html.Button(id='train-button-state', n_clicks=0, children='Train'),
            html.Div(id="train-output"),
            html.Button(id='reset-state', n_clicks=0, children='Reset Data'),
            html.Div(id="reset-output")
        ])


@app.callback(Output('reset-output', 'children'),
              [Input('reset-state', 'n_clicks')])
def reset(n_clicks):
    global data
    if n_clicks >= 1:
        data = reset_data()
        return ('Data reset')


@app.callback([Output('train-output', 'children'),
               Output('store-nclicks', 'children')],
              [Input('train-button-state', 'n_clicks')])
def update_n_clicks(n_clicks):
    if n_clicks == 1:
        return ('Model Training with {} datapoints'.format(data.X.shape[0]), n_clicks)
    elif n_clicks > 1:
        return ('Model Already Trained with {} datapoints'.format(data.X.shape[0]), n_clicks)
    else:
        return ('Start training', n_clicks)



@app.callback(Output('embedding', 'figure'),
              [Input('store-nclicks', 'children')],
              [State('al-strategy', 'value')])
def update_train(n_clicks, strategy):
    global data, train_obj, EMB_HISTORY
    print('hidden callback n_clicks {}'.
          format(n_clicks))
    if n_clicks == 1:
        train_obj = train_model()
        (X_TOLB, X_NOLB) = data_to_label(strategy)
        data.update_nolabel(X_NOLB)
        data.update_tolabel(X_TOLB)
        # print('x nolb shape {}'.format(data.X_NOLB.shape))
        train_obj.update_data(data)
        print('train obj x nolb shape {}'.format(train_obj.data.X_NOLB.shape))        
        #embeddings = train_obj.get_test_embedding()
        embeddings_tr = train_obj.get_trained_embedding()
        # embeddings_nolb = train_obj.get_nolb_embedding()
        embeddings_tolb = train_obj.get_tolb_embedding()
        #embeddings = np.concatenate((embeddings_tr, embeddings_nolb), axis=0)
        embeddings = np.concatenate((embeddings_tr, embeddings_tolb), axis=0)
        labels = np.concatenate((data.Y.numpy(),
                                 np.ones(embeddings_tolb.shape[0])*15),
                                axis=0)
        # np.random.seed(42)
        # reducer = umap.UMAP(random_state=42)
        umap_embeddings = reducer.fit_transform(embeddings)
        EMB_HISTORY = (umap_embeddings, labels)
        print('umap x{} y{}'.format(umap_embeddings[0,0], umap_embeddings[0,1]))
        fig = px.scatter(x=umap_embeddings[:, 0],
                         y=umap_embeddings[:, 1],
                         color=labels)
        # hover_data=[y_label])

    elif n_clicks > 1:
        '''
        embeddings_tr = train_obj.get_trained_embedding()
        embeddings_tolb = train_obj.get_tolb_embedding()
        embeddings = np.concatenate((embeddings_tr, embeddings_tolb), axis=0)
        labels = np.concatenate((data.Y.numpy(),
                                 np.ones(embeddings_tolb.shape[0])*15),
                                axis=0)
        umap_embeddings = reducer.fit_transform(embeddings)
        '''
        umap_embeddings = EMB_HISTORY[0]
        labels = EMB_HISTORY[1]
        print('umap x{} y{}'.format(umap_embeddings[0,0], umap_embeddings[0,1]))
        fig = px.scatter(x=umap_embeddings[:, 0],
                         y=umap_embeddings[:, 1],
                         color=labels)
        
        # hover_data=[y_label])

    else:
        if EMB_HISTORY is not None:
            umap_embeddings = EMB_HISTORY[0]
            labels = EMB_HISTORY[1]
            print('umap x{} y{}'.format(umap_embeddings[0,0], umap_embeddings[0,1]))
            fig = px.scatter(x=umap_embeddings[:, 0],
                             y=umap_embeddings[:, 1],
                             color=labels)
        else:
            x = np.random.rand(1000)
            y = np.random.rand(1000)

            fig = px.scatter(x=x, y=y)

    fig.update_layout(transition_duration=500,
                      yaxis=dict(scaleanchor='x',
                                 scaleratio=1,
                      ))
    return fig


@app.callback(Output('number-output', 'children'),
              [Input('submit-button-state', 'n_clicks')],
              [State('label0', 'value'),
               State('label1', 'value'),
               State('label2', 'value'),
               State('label3', 'value'),
               State('label4', 'value'),
               State('label5', 'value'),
               State('label6', 'value'),
               State('label7', 'value'),
               State('label8', 'value'),
               State('label9', 'value')])
def update_output(n_clicks, input0, input1, input2, input3,
                  input4, input5, input6, input7, input8, input9):
    global data
    if n_clicks == 1: 
        Y_TOLB = torch.tensor([input0, input1, input2,
                               input3, input4, input5,
                               input6, input7, input8, input9])

        X = torch.cat([data.X, X_TOLB], dim=0)
        Y = torch.cat([data.Y, Y_TOLB], dim=0)
        data.update_data(X, Y)
        return u'''
        Labels submitted,
        Training dataset has {} datapoints
        '''.format(X.shape[0])


if __name__ == '__main__':
    # strategy = EntropySampling(X, Y, X_TE, Y_TE, X_NOLB, net, handler, args)
    app.run_server(port=os.getenv("CDSW_APP_PORT"))
