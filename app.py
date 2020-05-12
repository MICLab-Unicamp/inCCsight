
import os
import pathlib
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

import inputfuncs
import ccprocess

import segmfuncs
import parcelfuncs

import webbrowser
from threading import Timer

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

# GENERAL DEFINITIONS -------------------------------------------------------------------------

segmentation_methods_dict = {'ROQS': segmfuncs.segm_roqs,
                             'Watershed': segmfuncs.segm_roqs}

parcellation_methods_dict = {'Witelson': parcelfuncs.parc_witelson,
                             'Hofer': parcelfuncs.parc_hofer,
                             'Chao': parcelfuncs.parc_chao,
                             'Cover': parcelfuncs.parc_cover,
                             'Freesurfer': parcelfuncs.parc_freesurfer}

# DATA IMPORTING -----------------------------------------------------------------------------

# Arg parser
opts = inputfuncs.get_parser().parse_args()

# Get subjects and assemble groups considering parent folders
path_dict = {}
if opts.dirs is not None:
    for directory in opts.dirs:
        if directory is not None:
            if inputfuncs.check_directory(directory, opts.basename):
                path_dict[directory.rsplit('/', 2)[1]] = directory

group_dict = {}
if opts.parents is not None:
    for parent in opts.parents:
        if parent is not None:
            
            directory_dict = inputfuncs.import_parent(parent, opts.basename)
            
            path_dict.update(directory_dict)
            group_dict[parent.rsplit('/', 1)[1]] = list(directory_dict.keys())
            print('DIRECTORY',parent.rsplit('/', 1)[1])
print('groupdict',group_dict.keys())


# DATA PROCESSING -----------------------------------------------------------------------------

# Create dataframe for each segmentation method

scalar_statistics_dict = {}
scalar_statistics_names = ['Group','FA','FA StdDev','MD','MD StdDev','RD','RD StdDev','AD','AD StdDev']

# Segment and get info

for subject_name, subject_path in tqdm(path_dict.items()):

    for segmentation_method in segmentation_methods_dict.keys():

        if segmentation_method in opts.segm:

            _, _, scalar_statistics, _ = ccprocess.segment(subject_path, 
                                                           segmentation_method, 
                                                           segmentation_methods_dict, 
                                                           parcellation_methods_dict, 
                                                           opts.basename)

            for group_name, group_list in group_dict.items():
                if subject_name in group_list:
                    scalar_statistics = [group_name] + list(scalar_statistics)


            if segmentation_method not in scalar_statistics_dict.keys(): 
                scalar_statistics_dict[segmentation_method] = {subject_name: list(scalar_statistics)}
            else:
                scalar_statistics_dict[segmentation_method][subject_name] = list(scalar_statistics)

# Transform info to pandas dataframes

for segmentation_method in segmentation_methods_dict.keys():
    scalar_statistics_dict[segmentation_method] = pd.DataFrame.from_dict(scalar_statistics_dict[segmentation_method], 
                                                                         orient='index', 
                                                                         columns=scalar_statistics_names)


# VISUALIZATION -------------------------------------------------------------------------------

app = dash.Dash(__name__, 
               meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
               external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app.server
app.config["suppress_callback_exceptions"] = True

def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Img(src=app.get_asset_url("unicampw.png")),
            html.Img(src=app.get_asset_url("miclab.png")),
            html.H5("Corpus Callosum DTI", className='cctitle'),
            html.H5("Data Explorer", className='detitle'),
        ],
    )


def build_graph_title(title):
    return html.P(className="graph-title", children=title)

def build_segm_scatterplot():

    df = scalar_statistics_dict['ROQS']
    fig = px.scatter(df, 
                    x="FA", 
                    y="AD", 
                    color="Group", 
                    marginal_y="violin", 
                    marginal_x="histogram",
                    hover_name=df.index)
    return fig

def build_segm_scattermatrix():

    df = scalar_statistics_dict['ROQS']
    fig = px.scatter_matrix(df, 
                            dimensions=['FA','MD','RD','AD'],
                            color='Group',
                            hover_name=df.index)
    return fig



app.layout = html.Div(
    children=[
        html.Div(
            id="top-row",
            children=[
                html.Div(
                    className="row",
                    id="top-row-header",
                    children=[
                        html.Div(
                            id="header-container",
                            children=[
                                build_banner(),
                                html.H5(
                                    children=["This is an data exploration and visualization tool for diffusion tensor images of the corpus callosum.",
                                              " Upload data folders to begin. Further information can be found here (link)."
                                            ],
                                    id="instruct",
                                ),
                            ],
                        )
                    ],
                ),

            # Top row right
                html.Div(
                    className="row",
                    id="top-row-graphs",
                    children=[
                        dbc.Row([

                        #Subjects list
                            dbc.Col([
                                html.Div(
                                    id="subject-list-container",
                                    children=[
                                        build_graph_title("Subjects"),
                                        html.Div(                                              
                                             dash_table.DataTable(
                                                id='table-subjects',
                                                columns=(
                                                    [{'id': 'Subjects', 
                                                      'name': 'Subjects'}]
                                                ),
                                                data=[{"Subjects": i} for i in list(path_dict.keys())],
                                                style_header={'display':'none'},
                                                style_table={'overflowY': 'scroll',
                                                             'overflowX': 'hidden', 
                                                             'height': '500px',
                                                             'width': '250px',
                                                             'border-width': '1px',
                                                             'border-style': 'solid',
                                                             'border-radius': '5px',
                                                             'border-color': 'white'},
                                                style_data_conditional=[{
                                                    'if':{'row_index':'odd'}, 'backgroundColor':'rgb(60,60,70)'}],
                                                style_cell={'fontSize': 21,
                                                            'border': 'none',
                                                            'border-radius':'5px', 
                                                            'font-family': 'Open Sans',
                                                            'textAlign': 'center',
                                                            'color':'rgb(255,255,255)',
                                                            'backgroundColor':'rgb(50,50,70)'},
                                            ), id='subject-table-container',
                                        )
                                    ],
                                ),
                            ], width = True),

                        # Create buttons
                            dbc.Col([

                                dbc.Button('>>', 
                                    id='group-button',
                                    style={'margin':'250px -25px 0 15px',
                                           'verticalAlign':'middle',
                                           'color':'white',
                                           'line-width': '1px'},
                                    size='lg',
                                    outline=False,
                                ),
                            
                            ], width='auto'),

                        # Groups list
                            dbc.Col([
                                html.Div(
                                    id="group-list-container",
                                    children=[
                                        html.Div(
                                            id="group-list-header",
                                            children=[
                                                build_graph_title("Groups"),
                                                html.Div(                                              
                                                    dash_table.DataTable(
                                                        id='table-groups',
                                                        columns=(
                                                            [{'id': 'Groups', 
                                                              'name': 'Groups', 
                                                              'selectable': True}]
                                                        ),
                                                        data = [{"Groups": i} for i in list(group_dict.keys())],
                                                        style_header={'display':'none'},
                                                        style_table={'overflowY': 'scroll',
                                                                     'overflowX': 'hidden', 
                                                                     'height': '500px',
                                                                     'width': '250px',
                                                                     'border-width': '1px',
                                                                     'border-style': 'solid',
                                                                     'border-radius': '5px',
                                                                     'border-color': 'white'},
                                                        style_data_conditional=[{
                                                            'if':{'row_index':'odd'}, 'backgroundColor':'rgb(60,60,70)'}],
                                                        style_cell={'fontSize': 21,
                                                                    'border': 'none',
                                                                    'border-radius':'5px', 
                                                                    'font-family': 'Open Sans',
                                                                    'textAlign': 'center',
                                                                    'color':'rgb(255,255,255)',
                                                                    'backgroundColor':'rgb(50,50,70)'},
                                                        #editable = True
                                                    )
                                            )

                                            ],
                                        ),
                                    ],
                                ),
                            ], width = True),


                        # Quality control
                            dbc.Col([
                                html.Div(
                                    id='quality-container',
                                    children=[
                                        build_graph_title("Quality Check"),
                                        dcc.Tabs([
                                            dcc.Tab(label='ROQS', children='Oi bonita', className='tabclass', selected_className='tabclass-selected'),
                                            dcc.Tab(label='Watershed', children='Oi bonita2', className='tabclass', selected_className='tabclass-selected'),
                                            dcc.Tab(label='STAPLE', children='Oi bonita3', className='tabclass', selected_className='tabclass-selected'),
                                            dcc.Tab(label='Mask', children='Oi bonita4', className='tabclass', selected_className='tabclass-selected'),
                                        ])
                                    ], style={'marginLeft':'4rem',
                                              'flexGrow':1,
                                              'width':'700px'}
                                )
                            ], width=True)

                        ], id='top-row-bootstrap', justify='end')

                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            id="bottom-row",
            children=[
                # Scatter boxplot
                html.Div(
                    id="form-bar-container",
                    className="six columns",
                    children=[
                        build_graph_title("Scatter Matrix"),
                        dcc.Graph(id="scatter_matrix", figure=build_segm_scattermatrix()),
                    ],
                ),
                html.Div(
                    # Selected well productions
                    id="well-production-container",
                    className="six columns",
                    children=[
                        build_graph_title("Scatter Plot"),
                        dcc.Graph(id="scatter_plot", figure=build_segm_scatterplot()),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            id="second-bottom-row",
            children=[
                # Scatter boxplot
                html.Div(
                    id="graph-container",
                    className="six columns",
                    children=[
                        build_graph_title(""),
                        dcc.Graph(id="oto_grapho", figure=build_segm_scattermatrix()),
                    ],
                ),
                html.Div(
                    # Selected well productions
                    id="boxplot-container",
                    className="six columns",
                    children=[
                        build_graph_title("Box Plot"),
                        #dcc.Graph(id="boxplot", figure=build_segm_boxplot()),
                    ],
                ),
            ],
        ),
    ]
)


# Add group
@app.callback(
    [Output("table-groups", "data"),
     Output("table-subjects", "selected_cells")],
    [Input("group-button", "n_clicks")],
    [State("table-subjects","selected_cells")])
def update_groups(n_clicks, selected_cells):
    subject_list = list(path_dict.keys())
    selected_subjects = []
    
    for i, cell_dict in enumerate(selected_cells):
        subject_key = subject_list[cell_dict['row']]
        selected_subjects.append(subject_key)
    
    new_group_name = "New Group"
    k = 1
    while new_group_name in group_dict.keys():
        new_group_name = "New Group " + str(k)
        k += 1

    group_dict[new_group_name] = selected_subjects

    return [{"Groups": i} for i in list(group_dict.keys())], {}
    

# Update bar plot
@app.callback(Output("scatter_matrix", "figure"),
    [Input("scatter_matrix", "selectedData")])
def update_scattermatrix(data):
    return {fig: build_segm_scattermatrix()}



# SERVER CONFIG ---------------------------------------------------------------------------------

#port = 5000 + random.randint(0, 999)
port = 5050

def open_browser():
    url = "http://127.0.0.1:{0}".format(port)
    webbrowser.open(url)

if __name__ == "__main__":
    Timer(1.25, open_browser).start()
    app.run_server(debug=False, port=port)
