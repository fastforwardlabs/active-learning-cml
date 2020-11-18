
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

train_obj = train_model()
strategy = "entropy"
(X_TOLB, X_NOLB) = data_to_label(strategy)
data.update_nolabel(X_NOLB)
data.update_tolabel(X_TOLB)
print('x nolb shape {}'.format(data.X_NOLB.shape))
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

print(type(umap_embeddings))

import pandas as pd

df = pd.DataFrame(data=umap_embeddings, columns=["dim1", "dim2"])
print(df)
df['labels'] = labels
print(df)
groups = df.groupby("labels")
groups

for idx, val in groups:
  print(idx, val)
  
'''
fig = px.scatter(x=umap_embeddings[:, 0], 
                 y=umap_embeddings[:, 1],
                 color=labels)
fig.show()
'''

'''
import matplotlib.pyplot as plt
colors = ['violet', 'blue', 'red', 'yellow', 'green', 'orange', 'magenta', 'xkcd:lilac', 'xkcd:purplish blue', 'xkcd:light blue', 'xkcd: black', 'xkcd: black', 'xkcd: black', 'xkcd: black', 'xkcd: black', 'xkcd:fuchsia']
values = np.unique(labels)
colorvalues = [colors[int(l)] for l in labels]

#unique(colorvalues)

fig, ax = plt.subplots()


plt.scatter(x=umap_embeddings[:, 0], 
           y=umap_embeddings[:, 1],
           c=colorvalues,
           label=[str(int(l))+":"+colors[int(l)] for l in values]
          )
#plt.legend(values, [colors[int(l)] for l in values])
plt.show()

labels




embeddings_tolb.shape

np.unique(data.Y)
np.unique(labels)


np.ones(embeddings_tolb.shape[0])*15

#c = np.arange(1.0, len(np.unique(labels)))
c = np.random.randint(1, 5, size=10)
c
fig, ax = plt.subplots()

scatter = ax.scatter(x=umap_embeddings[:, 0], 
            y=umap_embeddings[:, 1], c=c)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
'''