import base64
import io
import pathlib

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output
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
                                        name="Perplexity",
                                        short="perplexity",
                                        min=3,
                                        max=100,
                                        step=None,
                                        val=30,
                                        marks={i: str(i) for i in [3, 10, 30, 50, 100]},
                                    ),
                                    NamedSlider(
                                        name="Initial PCA Dimensions",
                                        short="pca-dimension",
                                        min=25,
                                        max=100,
                                        step=None,
                                        val=50,
                                        marks={i: str(i) for i in [25, 50, 100]},
                                    ),
                                    NamedSlider(
                                        name="Learning Rate",
                                        short="learning-rate",
                                        min=10,
                                        max=200,
                                        step=None,
                                        val=100,
                                        marks={i: str(i) for i in [10, 50, 100, 200]},
                                    ),
                                    html.Div(
                                        html.Button(id="train", children=["Train"])
                                    ),
                                    html.Div(
                                        id="div-wordemb-controls",
                                        style={"display": "none"},
                                        children=[
                                            NamedInlineRadioItems(
                                                name="Display Mode",
                                                short="wordemb-display-mode",
                                                options=[
                                                    {
                                                        "label": " Regular",
                                                        "value": "regular",
                                                    },
                                                    {
                                                        "label": " Top-100 Neighbors",
                                                        "value": "neighbors",
                                                    },
                                                ],
                                                val="regular",
                                            ),
                                            dcc.Dropdown(
                                                id="dropdown-word-selected",
                                                placeholder="Select word to display its neighbors",
                                                style={"background-color": "#f2f3f4"},
                                            ),
                                        ],
                                    ),
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-3d-plot-tsne", 
                                      #style={"height": "98vh"}
                                      style={"height": 700}
                                     )
                        ],
                    ),
                    html.Div(
                        className="three columns",
                        id="euclidean-distance",
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
    '''
    @app.callback(
        Output("graph-3d-plot-tsne", "figure"),
        [
            Input("strategy", "value"),
            Input("train", "n_clicks"),
        ],
    )
    def train_display_3d_scatter_plot(strategy, n_clicks):
        if n_clicks >= 1:

            # Plot layout
            axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

            
            layout = go.Layout(
                showlegend=True,
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis = dict(autorange=True,
                             showgrid=False,
                             showline=False,
                             ticks='',
                             showticklabels=False),
                yaxis = dict(autorange=True,
                             showgrid=False,
                             showline=False,
                             ticks='',
                             showticklabels=False))

            global data, train_obj, EMB_HISTORY, embedding_df, orig_x
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
            # np.random.seed(42)
            # reducer = umap.UMAP(random_state=42)
            umap_embeddings = reducer.fit_transform(embeddings)
            EMB_HISTORY = (umap_embeddings, labels)
            print('umap x{} y{}'.format(umap_embeddings[0,0], umap_embeddings[0,1]))
            
            embedding_df = pd.DataFrame(data=umap_embeddings, columns=["dim1", "dim2"])
            #print(df)
            embedding_df['labels'] = labels_text
            groups = embedding_df.groupby("labels")

            figure = generate_figure_image(groups, layout)
            return figure
      
    '''
    @app.callback(
        Output("graph-3d-plot-tsne", "figure"),
        [
            Input("strategy", "value"),
            Input("slider-epochs", "value"),
            Input("slider-perplexity", "value"),
            Input("slider-pca-dimension", "value"),
            Input("slider-learning-rate", "value"),
            Input("dropdown-word-selected", "value"),
            Input("radio-wordemb-display-mode", "value"),
        ],
    )
    def display_3d_scatter_plot(
        strategy,
        epochs,
        perplexity,
        pca_dim,
        learning_rate,
        selected_word,
        wordemb_display_mode,
    ):
        if strategy:

            # Plot layout
            axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

            
            layout = go.Layout(
                showlegend=True,
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis = dict(autorange=True,
                             showgrid=False,
                             showline=False,
                             ticks='',
                             showticklabels=False),
                yaxis = dict(autorange=True,
                             showgrid=False,
                             showline=False,
                             ticks='',
                             showticklabels=False)
                #margin=dict(l=0, r=0, b=0, t=0),
                #scene=dict(xaxis=axes, yaxis=axes),
            )

            global data, train_obj, EMB_HISTORY, embedding_df, orig_x
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
            # np.random.seed(42)
            # reducer = umap.UMAP(random_state=42)
            umap_embeddings = reducer.fit_transform(embeddings)
            EMB_HISTORY = (umap_embeddings, labels)
            print('umap x{} y{}'.format(umap_embeddings[0,0], umap_embeddings[0,1]))
            
            embedding_df = pd.DataFrame(data=umap_embeddings, columns=["dim1", "dim2"])
            #print(df)
            embedding_df['labels'] = labels_text
            groups = embedding_df.groupby("labels")

            figure = generate_figure_image(groups, layout)
            return figure
    
    @app.callback(
        Output("div-plot-click-image", "children"),
        [
            Input("graph-3d-plot-tsne", "clickData"),
            Input("strategy", "value"),
            Input("slider-iterations", "value"),
            Input("slider-perplexity", "value"),
            Input("slider-pca-dimension", "value"),
            Input("slider-learning-rate", "value"),
        ],
    )
    def display_click_image(
        clickData, strategy, iterations, perplexity, pca_dim, learning_rate
    ):
        print("its here")
        if clickData:
            # Load the same dataset as the one displayed
            '''
            try:
                data_url = [
                    "demo_embeddings",
                    str(dataset),
                    "iterations_" + str(iterations),
                    "perplexity_" + str(perplexity),
                    "pca_" + str(pca_dim),
                    "learning_rate_" + str(learning_rate),
                    "data.csv",
                ]

                full_path = PATH.joinpath(*data_url)
                embedding_df = pd.read_csv(full_path, encoding="ISO-8859-1")

            except FileNotFoundError as error:
                print(
                    error,
                    "\nThe dataset was not found. Please generate it using generate_demo_embeddings.py",
                )
                return
            '''
            # Convert the point clicked into float64 numpy array
            click_point_np = np.array(
                [clickData["points"][0][i] for i in ["dim1", "dim2"]]
            ).astype(np.float64)
            print(click_point_np)
            # Create a boolean mask of the point clicked, truth value exists at only one row
            bool_mask_click = (
                embedding_df.loc[:, "dim1":"dim2"].eq(click_point_np).all(axis=1)
            )
            print(bool_mask_click)
            # Retrieve the index of the point clicked, given it is present in the set
            if bool_mask_click.any():
                clicked_idx = embedding_df[bool_mask_click].index[0]

                # Retrieve the image corresponding to the index
                #image_vector = data_dict[dataset].iloc[clicked_idx]
                image_vector = orig_x.iloc[clicked_idx]
                image_np = image_vector.values.reshape(28, 28).astype(np.float64)

                # Encode image into base 64
                image_b64 = numpy_to_b64(image_np)

                return html.Img(
                    src="data:image/png;base64, " + image_b64,
                    style={"height": "25vh", "display": "block", "margin": "auto"},
                )
        return None
            
    @app.callback(
        Output("div-plot-click-message", "children"),
        [Input("graph-3d-plot-tsne", "clickData"), Input("strategy", "value")],
    )
    def display_click_message(clickData, strategy):
        # Displays message shown when a point in the graph is clicked, depending whether it's an image or word
        if strategy in strategy_names:
            if clickData:
                print(clickData)
                # Convert the point clicked into float64 numpy array
                click_point_np = np.array(
                    [clickData["points"][0][i] for i in ["x", "y"]]
                ).astype(np.float64)
                print(click_point_np)
                # Create a boolean mask of the point clicked, truth value exists at only one row
                bool_mask_click = (
                    embedding_df.loc[:, "dim1":"dim2"].eq(click_point_np).all(axis=1)
                )
                print(bool_mask_click)
                # Retrieve the index of the point clicked, given it is present in the set
                if bool_mask_click.any():
                    clicked_idx = embedding_df[bool_mask_click].index[0]

                    # Retrieve the image corresponding to the index
                    #image_vector = data_dict[dataset].iloc[clicked_idx]
                    image_vector = orig_x[clicked_idx]
                    print(image_vector.shape)
                    image_np = image_vector.reshape(28, 28).astype(np.float64)

                    # Encode image into base 64
                    image_b64 = numpy_to_b64(image_np)
                    print(image_b64)
                    return html.Img(
                        #src="data:image/png;base64, " + image_b64,
                        src='data:image/png;base64,{}'.format(image_b64),
                        style={"height": "25vh", "display": "block", "margin": "auto"},
                    )

                return "Image Selected"
            else:
                return "Click a data point on the scatter plot to display its corresponding image."
