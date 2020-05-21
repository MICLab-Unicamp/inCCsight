
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

scalar_list = ['FA', 'MD', 'RD', 'AD']
colors_list = px.colors.qualitative.Plotly

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

# Import the subjects inside the parents folders
group_dict = {}
if opts.parents is not None:
    for parent in opts.parents:
        if parent is not None:
            
            directory_dict = inputfuncs.import_parent(parent, opts.basename)
            path_dict.update(directory_dict)

            # Create dict with subjects as keys and group (parents names) as values
            group_dict.update(dict.fromkeys(directory_dict.keys(), [parent.rsplit('/', 1)[1]]))
            

# DATA PROCESSING -----------------------------------------------------------------------------

# Create dataframe for each segmentation method

scalar_statistics_dict = {}
scalar_midlines_dict = {}

scalar_statistics_names = ['FA','FA StdDev','MD','MD StdDev','RD','RD StdDev','AD','AD StdDev']
scalar_midline_names = list(range(0,200))

# Segment and get info

for subject_name, subject_path in tqdm(path_dict.items()):

    for segmentation_method in segmentation_methods_dict.keys():

        if segmentation_method in opts.segm:

            _, _, scalar_statistics, scalar_midlines ,_ = ccprocess.segment(subject_path, 
                                                                           segmentation_method, 
                                                                           segmentation_methods_dict, 
                                                                           parcellation_methods_dict, 
                                                                           opts.basename)

            if segmentation_method not in scalar_statistics_dict.keys(): 
                scalar_statistics_dict[segmentation_method] = {subject_name: list(scalar_statistics)}
                
                scalar_midlines_dict[segmentation_method] = {'FA':{},'MD':{},'RD':{},'AD':{}}
                for scalar in scalar_list:
                    scalar_midlines_dict[segmentation_method][scalar] = {subject_name: list(scalar_midlines[scalar])}

            else:
                scalar_statistics_dict[segmentation_method][subject_name] = list(scalar_statistics)
                
                for scalar in scalar_list:
                    scalar_midlines_dict[segmentation_method][scalar][subject_name] = list(scalar_midlines[scalar])


# Transform info to pandas dataframes

df_group = pd.DataFrame.from_dict(group_dict, orient='index', columns=["Group"])

for segmentation_method in segmentation_methods_dict.keys():
    scalar_statistics_dict[segmentation_method] = pd.DataFrame.from_dict(scalar_statistics_dict[segmentation_method], 
                                                                         orient='index', 
                                                                         columns=scalar_statistics_names)
    for scalar in scalar_list:
        scalar_midlines_dict[segmentation_method][scalar] = pd.DataFrame.from_dict(scalar_midlines_dict[segmentation_method][scalar], 
                                                                                   orient='index', 
                                                                                   columns=scalar_midline_names)



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

def build_segm_scatterplot(segmentation_method = 'ROQS', scalar_x = 'FA', scalar_y = 'MD'):

    df = pd.concat([df_group, scalar_statistics_dict[segmentation_method]], axis=1).reset_index()
    fig = px.scatter(df, 
                    x=scalar_x, 
                    y=scalar_y, 
                    color="Group", 
                    marginal_y="violin", 
                    marginal_x="histogram",
                    hover_name=df.index)
    fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)',legend_orientation="h")
    return fig

def build_segm_scattermatrix(segmentation_method = 'ROQS'):

    df = pd.concat([df_group, scalar_statistics_dict[segmentation_method]], axis=1).reset_index()
    fig = px.scatter_matrix(df, 
                            dimensions=['FA','MD','RD','AD'],
                            color='Group',
                            hover_name=df.index)
    fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)',legend_orientation="h")
    return fig

def build_midline_plot(segmentation_method='ROQS', scalar='FA'):

    df = pd.concat([df_group, scalar_midlines_dict[segmentation_method][scalar]], axis=1).reset_index()
    df_grouped = df.groupby('Group').mean().transpose()
    df_melt = pd.melt(df_grouped.reset_index(), id_vars='index', value_vars=set(np.hstack(list(group_dict.values()))))
    
    fig = px.line(df_melt, x='index', y='value', color='Group')
    fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h")    
    
    return fig

def build_segm_boxplot(scalar):

    df = pd.concat([df_group, scalar_statistics_dict[segmentation_method]], axis=1).reset_index()
    
    fig = px.box(df, y=scalar, color='Group', hover_name=df.index)
    fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)',legend_orientation="h")

    layout = dcc.Graph(id="boxplot"+scalar, figure=fig)

    return layout

def build_scalar_dropdown():

    options = [{'label': scalar, 'value': scalar} for scalar in scalar_list]

    layout = html.Div([
                html.Div([
                    dcc.Dropdown(id='dropdown-scalars-left',
                                 options=options,
                                 multi=False,
                                 value='FA',
                                 style={'float': 'right', 'width':'150px'}),
                    dcc.Dropdown(id='dropdown-scalars-right',
                                 options=options,
                                 multi=False,
                                 value='RD',
                                 style={'float': 'right', 'width':'150px', 'margin-left':'20px'})
                ], className='row', style={'justifyContent':'center'}),
            ]
        )
    return layout

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
                                    children=["This is a data exploration and visualization tool for diffusion tensor images of the corpus callosum.",
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

                        #Subjects list
                            html.Div([
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
                            ], className = "two columns"),

                        # Create buttons
                            html.Div([
                                html.Button('>>', 
                                    id='group-button',
                                ),                            
                            ], id = "button-container"),

                        # Groups list
                            html.Div([
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
                                                        data = [{"Groups": i} for i in set(np.hstack(list(group_dict.values())))],
                                                        style_header={'display':'none'},
                                                        style_table={'overflowY': 'scroll',
                                                                     'overflowX': 'hidden', 
                                                                     'height': '500px',
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
                            ], className = "two columns"),


                        # Quality control
                            html.Div([
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
                                    ]
                                )
                            ], className = "four columns")
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
                    className="four columns",
                    children=[
                        build_graph_title("Scatter Matrix"),
                        dcc.Graph(id="scatter_matrix", figure=build_segm_scattermatrix()),
                    ],
                ),
                # Selected well productions
                html.Div(
                    id="well-production-container",
                    className="four columns",
                    children=[
                        build_graph_title("Scatter Plot"),
                        dcc.Graph(id="scatter_plot"),
                        build_scalar_dropdown(),
                    ],
                ),
                # Midline plots
                html.Div(
                    id="midline-container",
                    className="four columns",
                    children=[
                        build_graph_title("Midline Plots"),
                        dcc.Graph(id="midline_graph", figure=build_midline_plot()),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            id="second-bottom-row",
            children=[
                html.Div(
                        id="boxplot-container1",
                        className="three columns",
                        children=[
                            build_graph_title("FA BoxPlot"),
                            build_segm_boxplot("FA"),
                        ],
                    ),
                html.Div(
                        id="boxplot-container2",
                        className="three columns",
                        children=[
                            build_graph_title("MD BoxPlot"),
                            build_segm_boxplot("MD"),
                        ],
                    ),
                html.Div(
                        id="boxplot-container3",
                        className="three columns",
                        children=[
                            build_graph_title("RD BoxPlot"),
                            build_segm_boxplot("RD"),
                        ],
                    ),
                html.Div(
                        id="boxplot-container4",
                        className="three columns",
                        children=[
                            build_graph_title("AD BoxPlot"),
                            build_segm_boxplot("AD"),
                        ],
                    ),
            ],
        ),
        html.Div(
            className="row",
            id="third-bottom-row",
            children=[
                html.Div(
                    # Selected well productions
                    id="othersss-container",
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

# Update scatter plot
@app.callback(
    Output("scatter_plot", "figure"),
    [Input("dropdown-scalars-right","value"),
     Input("dropdown-scalars-left","value")])
def update_scatterplot_axis(scalar_x, scalar_y):
    return build_segm_scatterplot(scalar_x=scalar_x, scalar_y=scalar_y)


# Add group
@app.callback(
    [Output("table-groups", "data"),
     Output("table-subjects", "selected_cells")],
    [Input("group-button", "n_clicks")],
    [State("table-subjects","selected_cells")])
def update_groups(n_clicks, selected_cells):
    
    new_group_name = "New Group"
    k = 1
    while new_group_name in set(np.hstack(list(group_dict.values()))):
        new_group_name = "New Group " + str(k)
        k += 1

    subject_list = list(path_dict.keys())

    if selected_cells is not None:
        for i, cell_dict in enumerate(selected_cells):
            subject_key = subject_list[cell_dict['row']]
            group_dict[subject_key] += [new_group_name] 

    df_group = pd.DataFrame.from_dict(group_dict, orient='index', columns=["Group"])

    return [{"Groups": i} for i in set(np.hstack(list(group_dict.values())))], {}



# SERVER CONFIG ---------------------------------------------------------------------------------

#port = 5000 + random.randint(0, 999)
port = 5050

def open_browser():
    url = "http://127.0.0.1:{0}".format(port)
    webbrowser.open(url)

if __name__ == "__main__":
    Timer(1.25, open_browser).start()
    app.run_server(debug=False, port=port)
