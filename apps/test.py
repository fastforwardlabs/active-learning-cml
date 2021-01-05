import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc

app = dash.Dash()

app.layout = html.Div([
    html.Button('Click Me', id='button'),
    html.H3(id='button-clicks'),

    html.Hr(),

    html.Label('Input 1'),
    dcc.Input(id='input-1'),

    html.Label('Input 2'),
    dcc.Input(id='input-2'),

    html.Label('Slider 1'),
    dcc.Slider(id='slider-1'),

    html.Button(id='button-2'),

    html.Div(id='output')
])

@app.callback(
    [
        Output('button-clicks', 'children'),
        Output('button-2', 'n_clicks'),
    ],
    [
        Input('button', 'n_clicks')
    ])
def clicks(n_clicks):
    return ['Button has been clicked {} times'.format(n_clicks), 0]

@app.callback(
    Output('output', 'children'),
    [Input('button-2', 'n_clicks')],
    state=[State('input-1', 'value'),
     State('input-2', 'value'),
     State('slider-1', 'value')])
def compute(n_clicks, input1, input2, slider1):
    if n_clicks == 0:
        return 'A computation based off of {}, and {}'.format(
            input1, input2)
    else:
        return 'A computation based off of {}, {}, and {}'.format(
            input1, input2, slider1)

if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(port=os.getenv("CDSW_APP_PORT"))