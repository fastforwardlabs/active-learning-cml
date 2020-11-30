import base64
import io
import pathlib

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import pandas as pd
import plotly.graph_objs as go
import scipy.spatial.distance as spatial_distance

from PIL import Image, ImageEnhance
from dataset import get_dataset, get_handler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from model import get_net
from data import Data
from train import Train
from sample import Sample
import torch
import umap
import os

dataset_name = 'MNIST'
strategy_names = ['random', 'entropy', 'entropydrop']
data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,),
                                                          (0.3081,))])
handler = get_handler(dataset_name)
n_classes = 10
n_epoch = 10
seed = 123

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
                style={"padding": "50px 45px"},
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
                                        placeholder="Select a strategy",
                                        value="random",
                                    ),
                                    NamedSlider(
                                        name="Training Sample Size",
                                        short="samplesize",
                                        min=100,
                                        max=1000,
                                        step=None,
                                        val=200,
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
                                            i: str(i) for i in [10, 20, 30, 40]
                                        },
                                    ),
                                    NamedSlider(
                                        name="Learning Rate",
                                        short="learning-rate",
                                        min=0.001,
                                        max=0.1,
                                        step=None,
                                        val=0.01,
                                        marks={i: str(i) for i in [0.001, 0.01, 0.1]},
                                    ),
                                    html.Div(
                                        html.Button(id="train", children=["Train"])
                                    ),
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-2d-plot-umap", 
                                      #style={"height": "98vh"}
                                      style={"height": 700, "width": 700}
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
                                        children=[
                                            dcc.Input(
                                                id="input1", 
                                                type="number", 
                                                min=0, max=9, 
                                                step=1, 
                                                placeholder="enter label",
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
                                                children=["SUBMIT"],
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
                                    html.Div(id="div-plot-label-message")
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
  
def demo_callbacks(app):
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
                marker=dict(size=5, symbol="circle"),
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
        Output("graph-2d-plot-umap", "figure"),
        [
            Input("strategy", "value"),
            Input("train", "n_clicks"),
        ],
    )
    def display_3d_scatter_plot(
        strategy,
        n_clicks,
    ):
        if strategy:

            # Plot layout
            axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

            
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
                #margin=dict(l=0, r=0, b=0, t=0),
                #scene=dict(xaxis=axes, yaxis=axes),
            )
            global data, train_obj, EMB_HISTORY, orig_x, umap_embeddings_random, labels_text
            orig_x = data.X.numpy()
            
            print(n_clicks)
            if n_clicks is not None:  
                if n_clicks >= 1:
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
                    labels_text = [str(int(item)) for item in labels]
                    labels_text = ["to label" if x == "15" else x for x in labels_text]
            
                    orig_x = np.concatenate((data.X.numpy(), data.X_TOLB.numpy()),axis=0)
                    umap_embeddings = reducer.fit_transform(embeddings)
                    EMB_HISTORY = (umap_embeddings, labels)
                    print('umap x{} y{}'.format(umap_embeddings[0,0], umap_embeddings[0,1]))
                '''
                elif n_clicks > 1:
                    umap_embeddings = EMB_HISTORY[0]
                    labels = EMB_HISTORY[1]
                    print('umap x{} y{}'.format(umap_embeddings[0,0], umap_embeddings[0,1]))
                    labels_text = [str(int(item)) for item in labels]
                    labels_text = ["to label" if x == "15" else x for x in labels_text]
                '''
            else: 
                
                if EMB_HISTORY is not None:
                    print("it is in else emb_history")
                    umap_embeddings = EMB_HISTORY[0]
                    labels = EMB_HISTORY[1]
                    print('umap x{} y{}'.format(umap_embeddings[0,0], umap_embeddings[0,1]))
                    labels_text = [str(int(item)) for item in labels]
                    labels_text = ["to label" if x == "15" else x for x in labels_text]
                
                else:
                    
                    x = np.random.rand(1000).reshape(1000, 1)
                    y = np.random.rand(1000).reshape(1000, 1)
                    umap_embeddings = np.concatenate((x, y), axis=1)
                    umap_embeddings_random = umap_embeddings
                    labels = data.Y.numpy()
                    labels_text = [str(int(item)) for item in labels]
                    #labels_text = ["to label" if x == "15" else x for x in labels_text]
                

            embedding_df = pd.DataFrame(data=umap_embeddings, columns=["dim1", "dim2"])
            #print(df)
            embedding_df['labels'] = labels_text
            groups = embedding_df.groupby("labels")

            figure = generate_figure_image(groups, layout)
            return figure
              
    @app.callback(
        [
            Output("div-plot-click-image", "children"),
            Output("div-label-controls", "children"),
        ],
        [
            Input("graph-2d-plot-umap", "clickData"),
            Input("strategy", "value"),
        ],
    )
    def display_click_image(clickData, strategy):
        print("its div-plot-click-image")
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
                #print(umap_embeddings)
                '''
                # Create a boolean mask of the point clicked, truth value exists at only one row
                bool_mask_click = (
                    umap_embeddings.loc[:, "dim1":"dim2"].eq(click_point_np).all(axis=1)
                )
                clicked_idx = umap_embeddings[bool_mask_click].index[0]
                print(bool_mask_click)
                '''
                clicked_idx = np.where((umap_embeddings == (click_point_np)).all(axis=1))[0][0]
                
                print(clicked_idx)
            else:
                umap_embeddings = umap_embeddings_random
                clicked_idx = np.where((umap_embeddings == (click_point_np)).all(axis=1))[0][0]
                print(clicked_idx)
                
            # Retrieve the index of the point clicked, given it is present in the set
            # if bool_mask_click.any():
            # if clicked_idx is not None:    

            # Retrieve the image corresponding to the index
            #image_vector = data_dict[dataset].iloc[clicked_idx]
            image_vector = orig_x[clicked_idx]
            #print(type(data.X_TOLB))
            X_tolb_index = None
            if labels_text[clicked_idx] == "to label":
                X_tolb_index = np.where((data.X_TOLB.numpy() == image_vector).all((1,2)))[0][0]
                print("X_tolb_index: ", X_tolb_index)
            image_np = image_vector.reshape(28, 28).astype(np.float64)

            # Encode image into base 64
            image_b64 = numpy_to_b64(image_np)
            print(labels_text[clicked_idx])
            if labels_text[clicked_idx] == "to label":
                print("within to label")
                return(
                    html.Div(
                        #id="div-plot-click-image",
                        html.Img(
                            src='data:image/png;base64,{}'.format(image_b64),
                            style={"height": "25vh", "display": "block", "margin": "auto"},
                        )
                    ),
                    html.Div(
                        #id="div-label-controls",
                        #style={"display": "block"},
                        style={"visibility": "visible"},
                        
                        children=[
                            dcc.Input(
                                id="input1",
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
                                    #"margin": "auto"
                                    
                                },
                            ),
                            html.Button(
                                id="submit-button", 
                                children=["Submit"],
                                style={
                                    "text-align": "center",
                                    "margin-top": "5px",
                                    "margin-left": "10px",
                                    "margin-right": "5px", 
                                    "margin-bottom": "10px",
                                    "font-weight": "bold",
                                    #"margin": "auto"
                                },
                            ),
                        ],
                    )
                )
            else:        
                return(
                    html.Div(
                        html.Img(
                            #src="data:image/png;base64, " + image_b64,
                            src='data:image/png;base64,{}'.format(image_b64),
                            style={"height": "25vh", "display": "block", "margin": "auto"},
                        ),
                    ),
                    html.Div(
                        #id="div-label-controls",
                        #style={"display": "none"},
                        style={"visibility": "hidden"},
                        
                        children=[
                            dcc.Input(
                                id="input1",
                                type="number", 
                                min=0, max=9, 
                                step=1, 
                                placeholder="enter label",
                                style={
                                    "text-align": "center",
                                    "margin-bottom": "7px",
                                    "font-weight": "bold",
                                },
                            ),
                            html.Button(
                                id="submit-button", 
                                children=["Submit"],
                                style={
                                    "text-align": "center",
                                    "margin-bottom": "7px",
                                    "font-weight": "bold",
                                },
                            ),
                        ],
                        
                    )
                )
        return (None, None)
            
    @app.callback(
        Output("div-plot-click-message", "children"),
        [
            Input("graph-2d-plot-umap", "clickData"), 
            Input("strategy", "value")
        ],
    )
    def display_click_message(clickData, strategy):
        # Displays message shown when a point in the graph is clicked, depending whether it's an image or word
        
        if clickData:
            return "Image Selected"
        else:
            return "Click a data point on the scatter plot to display its corresponding image."
    
    @app.callback(
        Output('div-plot-label-message', 'children'),
        [Input('submit-button', 'n_clicks')],
        [State('input1', 'value')]
    )
    def update_output(n_clicks, input1):
        global data
        if n_clicks == 1: 
            Y_TOLB = torch.tensor([input1])
            print(Y_TOLB)
            # how to get corresponding X_TOLB?
            print("data.X_TOLB: ", data.X_TOLB.shape)
            print("X_tolb_index: ", X_tolb_index)
        
            X = torch.cat([data.X, data.X_TOLB[X_tolb_index].reshape(1, data.X_TOLB[X_tolb_index].shape[0], data.X_TOLB[X_tolb_index].shape[1])], dim=0)
            Y = torch.cat([data.Y, Y_TOLB], dim=0)
            data.update_data(X, Y)
            return u'''
            Label submitted,
            training dataset has {} datapoints
            '''.format(X.shape[0])