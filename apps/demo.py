# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2020
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products 
#  made up of hundreds of individual components, each of which was 
#  individually copyrighted.  Each Cloudera open source product is a 
#  collective work under U.S. Copyright Law. Your license to use the 
#  collective work is as provided in your written agreement with  
#  Cloudera.  Used apart from the collective work, this file is 
#  licensed for your use pursuant to the open source license 
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute 
#  this code. If you do not have a written agreement with Cloudera nor 
#  with an authorized and properly licensed third party, you do not 
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED 
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO 
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND 
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU, 
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS 
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE 
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES 
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF 
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import base64
import io
import pathlib
import torch
import umap
import os
import datetime
import shutil
import pandas as pd
import plotly.graph_objs as go
import scipy.spatial.distance as spatial_distance
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html

from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from plotly import tools
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from torchvision import transforms

from activelearning.dataset import get_dataset, get_handler
from activelearning.model import get_net
from activelearning.data import Data
from activelearning.train import Train
from activelearning.sample import Sample

"""
Global variables
"""  
dataset_name = 'MNIST'
strategy_names = ['random', 'entropy', 'entropydrop']
data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,),
                                                          (0.3081,))])
# create relative models folder
if not os.path.exists("models"):
    os.mkdir("models")
model_dir = "models/model_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
handler = get_handler(dataset_name)
n_classes = 10 
seed = 123
prev_reset_clicks = 0 
prev_train_clicks = 0
prev_submit_clicks = 0

"""
Set random seeds for UMAP
Initialize UMAP reducer
"""
np.random.seed(seed)
torch.manual_seed(seed)
reducer = umap.UMAP(random_state=seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
_, _, X_INIT, Y_INIT = get_dataset(dataset_name)

X_TR, X_TE, Y_TR, Y_TE = train_test_split(X_INIT, Y_INIT,
                                          test_size=0.2,
                                          random_state=seed,
                                          shuffle=True)

"""
VARs to control embedding figures
"""
EMB_HISTORY = None

def reset_data():
    global data
    data = Data(X, Y, X_TE, Y_TE, X_NOLB, X_TOLB,
                data_transform, handler, n_classes)
    return data

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
    return im_b64

def create_img(arr, shape=(28, 28)):
    arr = arr.reshape(shape).astype(np.float64)
    image_b64 = numpy_to_b64(arr)
    return image_b64

# get relative data folder
PATH = pathlib.Path(__file__).parent
with open(PATH.joinpath("demo_intro.md"), "r") as file:
  demo_intro_md = file.read()
with open(PATH.joinpath("demo_description.md"), "r") as file:
  demo_description_md = file.read()

# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")

def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )


def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f"div-{short}",
        style={"display": "inline-block"},
        children=[
            f"{name}:",
            dcc.RadioItems(
                id=f"radio-{short}",
                options=options,
                value=val,
                labelStyle={"display": "inline-block", "margin-right": "7px"},
                style={"display": "inline-block", "margin-left": "7px"},
            ),
        ],
    )

def create_layout(app):
    # Actual layout of the app
    return html.Div(
        className="row",
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            html.Div(
                className="row header",
                id="app-header",
                style={"background-color": "#f9f9f9"},
                children=[
                    html.Div(
                        [
                            html.Img(
                                src=app.get_asset_url("ff-logo.png"),
                                className="logo",
                                id="plotly-image",
                            )
                        ],
                        className="three columns header_img",
                    ),
                    html.Div(
                        [
                            html.H3(
                                "Active Learner",
                                className="header_title",
                                id="app-title",
                            )
                        ],
                        className="nine columns header_title_container",
                    ),
                ],
            ),
            # Demo Description
            html.Div(
                className="row background",
                id="demo-explanation",
                #style={"padding": "50px 45px"},
                style={"padding": "10px 40px"},
                children=[
                    html.Div(
                        id="description-text", children=dcc.Markdown(demo_intro_md)
                    ),
                    html.Div(
                        html.Button(id="learn-more-button", children=["Learn More"])
                    ),
                ],
            ),
            # Body
            html.Div(
                className="row background",
                style={"padding": "10px"},
                children=[
                    html.Div(
                        className="three columns",
                        children=[
                            Card(
                                [
                                    dcc.Dropdown(
                                        id="strategy",
                                        searchable=False,
                                        clearable=False,
                                        options=[
                                            {
                                                "label": "Random",
                                                "value": "random",
                                            },
                                            {
                                                "label": "Entropy",
                                                "value": "entropy",
                                            },
                                            {
                                                "label": "Entropy with dropout",
                                                "value": "entropydrop",
                                            },
                                        ],
                                        #placeholder="Select a strategy",
                                        value="entropy",
                                        disabled=False
                                    ),
                                    NamedSlider(
                                        name="Initial labeled training examples",
                                        short="samplesize",
                                        min=100,
                                        max=1000,
                                        step=None,
                                        val=1000,
                                        marks={i: str(i) for i in [100, 500, 1000]},
                                    ),
                                    NamedSlider(
                                        name="Number Of Epochs",
                                        short="epochs",
                                        min=10,
                                        max=50,
                                        step=None,
                                        val=10,
                                        marks={
                                            i: str(i) for i in [5, 10, 20, 30]
                                        },
                                    ),
                                    NamedSlider(
                                        name="Learning Rate",
                                        short="lr",
                                        min=0.001,
                                        max=0.1,
                                        step=None,
                                        val=0.01,
                                        marks={i: str(i) for i in [0.001, 0.01, 0.1]},
                                    ),
                                    html.Div(
                                        id="div-parameter-buttons",
                                        children=[
                                            html.Button("Train", id="train-button", n_clicks=0),
                                            html.Button("Reset", id="reset-button", 
                                                style={
                                                    "margin-left": "10px",
                                                },
                                                n_clicks=0,
                                            )
                                        ],
                                          
                                    ),
                                    
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-2d-plot-umap", 
                                      style={"height": "78vh"}
                                     )
                        ],
                    ),
                    html.Div(
                        className="three columns",
                        id="label-images",
                        children=[
                            Card(
                                style={"padding": "5px"},
                                children=[
                                    html.Div(
                                        id="div-plot-click-message",
                                        style={
                                            "text-align": "center",
                                            "margin-bottom": "7px",
                                            "font-weight": "bold",
                                        },
                                    ),
                                    html.Div(id="div-plot-click-image"),
                                    html.Div(
                                        id="div-label-controls",
                                        #style={"display": "none"},
                                        style={"visibility": "hidden"},
                                        children = [
                                          dcc.Input(
                                              id="input-label",
                                              type="number", 
                                              min=0, max=9, 
                                              step=1, 
                                              placeholder=0,
                                              style={
                                                  "size": "120",
                                                  "text-align": "center",
                                                  "margin-top": "5px",
                                                  "margin-left": "50px",
                                                  "margin-right": "5px", 
                                                  "margin-bottom": "10px",
                                                  "font-weight": "bold",
                                              },
                                          ),
                                          html.Button(
                                              id="submit-button", 
                                              n_clicks=0,
                                              children=["Submit"],
                                              style={
                                                  "text-align": "center",
                                                  "margin-top": "5px",
                                                  "margin-left": "10px",
                                                  "margin-right": "5px", 
                                                  "margin-bottom": "10px",
                                                  "font-weight": "bold",
                                              },
                                          ),
                                      ],
                                      
                                    ),
                                    html.Div(
                                        id="div-plot-label-message",
                                        style={
                                            "text-align": "center",
                                            "margin-bottom": "7px",
                                            'color': 'blue', 
                                            'font-style': 'italic'
                                        },
                                    )
                                ],
                            )
                        ],
                    ),
                ],
            ),
            # train, test results
            html.Div(
                className="row background",
                style={"padding": "10px", "height": "25vh"},
                
                
                children=[
                    html.Div(
                        className="six columns",
                        children =[
                          dcc.Graph(
                              id="div-results-loss-graph",
                              style={'height': "25vh"},
                              #animate=True
                          ),
                          dcc.Interval( 
                              id = 'loss-graph-update', 
                              interval = 20000, 
                              n_intervals = 0
                          ), 
                        ]
                    ),
                    html.Div(
                        className="six columns",
                        children =[
                          dcc.Graph(
                              id="div-results-acc-graph",
                              style={'height': "25vh"},
                              #animate=True
                          ),
                          dcc.Interval( 
                              id = 'acc-graph-update', 
                              interval = 20000, 
                              n_intervals = 0
                          ), 
                          
                        ]
                    )
                ],              
                
            ),

        ],
    )
  
def demo_callbacks(app):
    '''
    generates the scatter plot based on embedding data
    '''
    def generate_figure_image(groups, layout):
        data = []
        for idx, val in groups:
            scatter = go.Scatter(
                name=idx,
                x=val["dim1"],
                y=val["dim2"],
                text=[idx for _ in range(val["dim1"].shape[0])],
                textposition="top center",
                mode="markers",
                marker=dict(size=7, symbol="circle"),
            )
            data.append(scatter)
        figure = go.Figure(data=data, layout=layout)
        return figure


    # Callback function for the learn-more button
    @app.callback(
        [
            Output("description-text", "children"),
            Output("learn-more-button", "children"),
        ],
        [Input("learn-more-button", "n_clicks")],
    )    
    def learn_more(n_clicks):
        # If clicked odd times, the instructions will show; else (even times), only the header will show
        if n_clicks is None:
            n_clicks = 0
        if (n_clicks % 2) == 1:
            n_clicks += 1
            return (
                html.Div(
                    style={"padding-right": "15%"},
                    children=[dcc.Markdown(demo_description_md)],
                ),
                "Close",
            )
        else:
            n_clicks += 1
            return (
                html.Div(
                    style={"padding-right": "15%"},
                    children=[dcc.Markdown(demo_intro_md)],
                ),
                "Learn More",
            )


    @app.callback(
        [
            Output("graph-2d-plot-umap", "figure"),
            Output("strategy", "disabled"),
            Output("slider-samplesize", "disabled"),
            Output("slider-epochs", "disabled"),
            Output("slider-lr", "disabled"),
            Output("graph-2d-plot-umap", "clickData"), 
            #Output('div-plot-label-message', 'children')
        ],
        [            
            Input("train-button", "n_clicks"),
            Input("strategy", "value"),
            Input("slider-samplesize", "value"),
            Input("slider-epochs", "value"),
            Input("slider-lr", "value")       
        ],
    )
    def display_3d_scatter_plot(        
        train_clicks,
        strategy,
        samplesize,
        epochs,
        lr
    ):        
        strategy_disabled = False
        samplesize_disabled = False 
        epochs_disabled = False 
        lr_disabled = False
        
        # Plot layout
        layout = go.Layout(
            showlegend=True,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis = dict(autorange=True,
                         showgrid=False,
                         showline=False,
                         zeroline=False,
                         ticks='',
                         showticklabels=False),
            yaxis = dict(autorange=True,
                         showgrid=False,
                         showline=False,
                         zeroline=False,
                         ticks='',
                         showticklabels=False),
            legend=dict(yanchor="top",
                        y=0.99,xanchor="left",
                        x=0.01)
        )
        global data, train_obj, EMB_HISTORY, orig_x, umap_embeddings_random, labels_text, prev_train_clicks
        print("train_clicks: ", train_clicks)
        prev_train_clicks = train_clicks
        if train_clicks > 0:  
            if train_clicks == 1: # and reset_click == 0: 
                # disable parameter components
                strategy_disabled = True
                samplesize_disabled = True 
                epochs_disabled = True 
                lr_disabled = True
                orig_x = torch.empty([samplesize, 28, 28], dtype=torch.uint8)
                '''
                training
                '''
                print("start training: ", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                # create model directory    
                train_obj = Train(net, handler, epochs, lr, data, model_dir)
                train_obj.train()
                (X_TOLB, X_NOLB) = data_to_label(strategy)
                data.update_nolabel(X_NOLB)
                data.update_tolabel(X_TOLB)
                train_obj.update_data(data)
                print('train obj x nolb shape {}'.format(train_obj.data.X_NOLB.shape))        
                embeddings_tr = train_obj.get_trained_embedding()
                embeddings_tolb = train_obj.get_tolb_embedding()
                embeddings = np.concatenate((embeddings_tr, embeddings_tolb), axis=0)
                labels = np.concatenate((data.Y.numpy(),
                                     np.ones(embeddings_tolb.shape[0])*15),
                                    axis=0)
                labels_text = [str(int(item)) for item in labels]
                labels_text = ["to label" if x == "15" else x for x in labels_text]

                orig_x = np.concatenate((data.X.numpy(), data.X_TOLB.numpy()),axis=0)
                umap_embeddings = reducer.fit_transform(embeddings)
                EMB_HISTORY = (umap_embeddings, labels)
                print("end training: ", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

            elif train_clicks > 1: # and reset_click == 0:
                # disable parameter components
                strategy_disabled = True
                samplesize_disabled = True 
                epochs_disabled = True 
                lr_disabled = True                
                '''
                training
                '''
                print("start training: ", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                train_obj.train()
                (X_TOLB, X_NOLB) = data_to_label(strategy)
                data.update_nolabel(X_NOLB)
                data.update_tolabel(X_TOLB)
                train_obj.update_data(data)
                print('train obj x nolb shape {}'.format(train_obj.data.X_NOLB.shape))        
                embeddings_tr = train_obj.get_trained_embedding()
                embeddings_tolb = train_obj.get_tolb_embedding()
                embeddings = np.concatenate((embeddings_tr, embeddings_tolb), axis=0)
                labels = np.concatenate((data.Y.numpy(),
                                     np.ones(embeddings_tolb.shape[0])*15),
                                    axis=0)
                labels_text = [str(int(item)) for item in labels]
                labels_text = ["to label" if x == "15" else x for x in labels_text]

                orig_x = np.concatenate((data.X.numpy(), data.X_TOLB.numpy()),axis=0)
                umap_embeddings = reducer.fit_transform(embeddings)
                EMB_HISTORY = (umap_embeddings, labels)
                print("end training: ", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        else: 
            if EMB_HISTORY is not None:
                # disable parameter components
                strategy_disabled = True
                samplesize_disabled = True 
                epochs_disabled = True 
                lr_disabled = True
                
                print("with embedding history")
                umap_embeddings = EMB_HISTORY[0]
                labels = EMB_HISTORY[1]
                print('umap x{} y{}'.format(umap_embeddings[0,0], umap_embeddings[0,1]))
                labels_text = [str(int(item)) for item in labels]
                labels_text = ["to label" if x == "15" else x for x in labels_text]
            else: 
                print("no embedding history")
                train_ratio = samplesize/X_TR.shape[0]
                print("train_ratio", train_ratio)
                X_NOLB, X, Y_NOLB, Y = train_test_split(X_TR, Y_TR,
                                    test_size=train_ratio,
                                    random_state=seed,
                                    shuffle=True)
                X_TOLB = torch.empty([10, 28, 28], dtype=torch.uint8)
                '''
                Make data and train objects
                '''
                data = Data(X, Y, X_TE, Y_TE, X_NOLB, X_TOLB, 
                            data_transform, 
                            handler, n_classes)     
                print("data.X: ", data.X.shape)
                x = np.random.rand(samplesize).reshape(samplesize, 1)
                y = np.random.rand(samplesize).reshape(samplesize, 1)
                umap_embeddings = np.concatenate((x, y), axis=1)
                umap_embeddings_random = umap_embeddings
                labels = data.Y.numpy()
                labels_text = [str(int(item)) for item in labels]
                orig_x = data.X.numpy()

        embedding_df = pd.DataFrame(data=umap_embeddings, columns=["dim1", "dim2"])
        embedding_df['labels'] = labels_text
        groups = embedding_df.groupby("labels")
        figure = generate_figure_image(groups, layout)        
        return [figure, strategy_disabled, samplesize_disabled, epochs_disabled, lr_disabled, None]


    @app.callback(
        [
            Output("train-button", "n_clicks")
        ],
        [
            Input("reset-button", "n_clicks")
        ]
    )
    def reset(
        reset_clicks
    ):
        print("reset_clicks: ", reset_clicks)
        global EMB_HISTORY, prev_reset_clicks
        if reset_clicks >= 1:
            # need to take care of training results
            if os.path.exists(model_dir):
                try:
                    shutil.rmtree(model_dir)
                except OSError as e:
                    print("Error: %s : %s" % (model_dir, e.strerror))

            EMB_HISTORY = None
            prev_reset_clicks = reset_clicks            
            return  [0]
        #print("prev_train_clicks: ", prev_train_clicks)
        return [prev_train_clicks]


    @app.callback(
        [
            Output("div-plot-click-image", "children"),
            Output("div-label-controls", "children"),
        ],
        [
            Input("graph-2d-plot-umap", "clickData")
        ]
    )
    def display_click_image(clickData):
        #print("its div-plot-click-image")
        global X_tolb_index
        if clickData:
            print(clickData)
            # Convert the point clicked into float64 numpy array
            click_point_np = np.array(
                [clickData["points"][0][i] for i in ["x", "y"]]
            ).astype(np.float64)
            print(click_point_np)
                
            clicked_idx = None
            if EMB_HISTORY is not None:
                umap_embeddings = EMB_HISTORY[0]
                clicked_idx = np.where((umap_embeddings == (click_point_np)).all(axis=1))[0][0]
                #print(clicked_idx)
            else:
                umap_embeddings = umap_embeddings_random
                clicked_idx = np.where((umap_embeddings == (click_point_np)).all(axis=1))[0][0]
                #print(clicked_idx)
                
            image_vector = orig_x[clicked_idx]
            X_tolb_index = None
            if labels_text[clicked_idx] == "to label":
                X_tolb_index = np.where((data.X_TOLB.numpy() == image_vector).all((1,2)))[0][0]
            image_np = image_vector.reshape(28, 28).astype(np.float64)

            # Encode image into base 64
            image_b64 = numpy_to_b64(image_np)
            print(labels_text[clicked_idx])
            if labels_text[clicked_idx] == "to label":
                print("within to label")
                return(
                    html.Div(
                        html.Img(
                            src='data:image/png;base64,{}'.format(image_b64),
                            style={"height": "25vh", "display": "block", "margin": "auto"},
                        )
                    ),
                    html.Div(
                        style={"visibility": "visible"},
                        children=[
                            dcc.Input(
                                id="input-label",
                                type="number", 
                                min=0, max=9, 
                                step=1, 
                                placeholder=0,
                                style={
                                    "size": "120",
                                    "text-align": "center",
                                    "margin-top": "5px",
                                    "margin-left": "50px",
                                    "margin-right": "5px", 
                                    "margin-bottom": "10px",
                                    "font-weight": "bold",
                                },
                            ),
                            html.Button(
                                id="submit-button", 
                                n_clicks=0,
                                children=["Submit"],
                                style={
                                    "text-align": "center",
                                    "margin-top": "5px",
                                    "margin-left": "10px",
                                    "margin-right": "5px", 
                                    "margin-bottom": "10px",
                                    "font-weight": "bold",
                                },
                            ),
                        ],
                    )
                )
            else:        
                return(
                    html.Div(
                        html.Img(
                            src='data:image/png;base64,{}'.format(image_b64),
                            style={"height": "25vh", "display": "block", "margin": "auto"},
                        ),
                    ),
                    html.Div(
                        style={"visibility": "hidden"},
                        children=[
                            dcc.Input(
                                id="input-label",
                                type="number", 
                                min=0, max=9, 
                                step=1, 
                                placeholder=0,
                                style={
                                    "size": "120",
                                    "text-align": "center",
                                    "margin-top": "5px",
                                    "margin-left": "50px",
                                    "margin-right": "5px", 
                                    "margin-bottom": "10px",
                                    "font-weight": "bold",
                                },
                            ),
                            html.Button(
                                id="submit-button", 
                                n_clicks=0,
                                children=["Submit"],
                                style={
                                    "text-align": "center",
                                    "margin-top": "5px",
                                    "margin-left": "10px",
                                    "margin-right": "5px", 
                                    "margin-bottom": "10px",
                                    "font-weight": "bold",
                                },
                            ),
                        ],
                    )
                )
        else:
            #print("no clickData")
            return (None, None)

  
    @app.callback(
        Output("div-plot-click-message", "children"),
        [
            Input("graph-2d-plot-umap", "clickData")
        ]
    )
    def display_click_message(clickData):
        # Displays message shown when a point in the graph is clicked
        if clickData:
            return "Image Selected"
        else:
            return "Click a data point on the scatter plot to display its corresponding image."

          
    @app.callback(
        Output('div-plot-label-message', 'children'),
        [
            Input('submit-button', 'n_clicks')
        ],
        [State('input-label', 'value')]
    )
    def update_output(submit_clicks, input_label):
        global data, labels_text, prev_submit_clicks
        #print("submit_clicks: ", submit_clicks)
        if submit_clicks > 0: 
            prev_submit_clicks = submit_clicks
            Y_TOLB = torch.tensor([input_label])
            print("y labeled: ", Y_TOLB)            
            print("data.X_TOLB: ", data.X_TOLB.shape)
            print("X_tolb_index: ", X_tolb_index)
            labels_text[X_tolb_index] = str(int(Y_TOLB))
            # how to get corresponding X_TOLB?
            X = torch.cat([data.X, data.X_TOLB[X_tolb_index].reshape(1, data.X_TOLB[X_tolb_index].shape[0], data.X_TOLB[X_tolb_index].shape[1])], dim=0)
            Y = torch.cat([data.Y, Y_TOLB], dim=0)
            data.update_data(X, Y)
            return u'''Training dataset has {} datapoints'''.format(data.X.shape[0])
        else:
            return u''' '''

    
    @app.callback(
        Output("div-results-loss-graph", "figure"),
        [
            Input('loss-graph-update', 'n_intervals')
        ]
    )
    def display_loss_results(n):
        if 'train_obj' in vars() or 'train_obj' in globals():
            step = list(train_obj.step)
            y_train = list(train_obj.train_loss)
            y_val = list(train_obj.val_loss)
        else:
            step = [1, 10]
            y_train = []
            y_val = []            
        layout = go.Layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            yaxis={"title": "cross entropy loss",},
            xaxis={
                "title": "epochs",
                "range": [min(step),max(step)]
            },
            legend=dict(yanchor="bottom", y=0.1, 
                        xanchor="left", x=0.01),
            paper_bgcolor="#f2f3f4"            
        )                
        # line charts
        trace_train = go.Scatter(
            x=step,
            y=y_train,
            mode="lines+markers",
            name="Training",
            line=dict(color="#636EFA"),
            showlegend=True,
        )
        trace_val = go.Scatter(
            x=step,
            y=y_val,
            mode="lines+markers",
            name="Validation",
            line=dict(color="#00CC96"),
            showlegend=True,
        )
        figure = go.Figure(data=[trace_train, trace_val], layout=layout)
        return figure

    
    @app.callback(
        Output("div-results-acc-graph", "figure"),
        [
            Input('acc-graph-update', 'n_intervals')
        ]
    )
    def display_acc_results(n):
        if 'train_obj' in vars() or 'train_obj' in globals():
            step = list(train_obj.step)
            y_train = list(train_obj.train_acc)
            y_val = list(train_obj.val_acc)
        else:
            step = [0, 10]
            y_train = []
            y_val = []          
        layout = go.Layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            yaxis={
                "title": "accuracy",
                "range": [0.0,1.0]
            },
            xaxis={
                "title": "epochs",
                "range": [min(step),max(step)]
            },
            legend=dict(yanchor="bottom", y=0.1, 
                        xanchor="left", x=0.01),
            paper_bgcolor="#f2f3f4"            
        )                
        # line charts
        trace_train = go.Scatter(
            x=step,
            y=y_train,
            mode="lines+markers",
            name="Training",
            line=dict(color="#636EFA"),
            showlegend=True,
        )
        trace_val = go.Scatter(
            x=step,
            y=y_val,
            mode="lines+markers",
            name="Validation",
            line=dict(color="#00CC96"),
            showlegend=True,
        )
        figure = go.Figure(data=[trace_train, trace_val], layout=layout)
        return figure