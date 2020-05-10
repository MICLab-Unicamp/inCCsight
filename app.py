import pathlib
import os

import pandas as pd
import numpy as np

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

import constants
import inputfuncs

# INITIALIZE ----------------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets = [dbc.themes.BOOTSTRAP],
)
server = app.server
app.config["suppress_callback_exceptions"] = True

# GENERAL ------------------------------------------------------------------------------------

segmentation_methods = ['ROQS', 'Watershed', 'STAPLE', 'Imported Mask']
parcellation_methods = ['Witelson', 'Hofer', 'Chao', 'Cover', 'Freesurfer']



# DATA IMPORTING -----------------------------------------------------------------------------

# Arg parser
opts = inputfuncs.get_parser().parse_args()

# Get paths and check if files exist
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
            
            path_dict = dict(path_dict.items() + directory_dict.items())
            group_dict[parent.rsplit('/', 2)[1]] = list(directory_dict.keys())

# Check if there are CCLab folders


# DATA PROCESSING -----------------------------------------------------------------------------





# Load data
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

df_dict = {}
df_roqs = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "df_roqs.csv")), index_col=0)
df_dict["Watershed"] = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "df_watershed.csv")), index_col=0)







# LAYOUT STUFF ----------------------------------------------------------------------------------

colormap = {}
for ind, segmentation in enumerate(df_dict.keys()):
    colormap[segmentation] = constants.colors[ind]


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

                                build_graph_title("Select segmentation methods"),
                                dcc.Dropdown(
                                    id='segmentation-drop',
                                    options=[
                                        {'label': 'ROQS', 'value': 'ROQS'},
                                        {'label': 'Watershed', 'value': 'Watershed'},
                                        {'label': 'STAPLE', 'value': 'STAPLE'}],
                                    style={'backgroundColor':'rgba(0,0,0,0)',
                                           'fontSize':'2rem',
                                           'lineHeight':'42px',
                                           'width':'470px',
                                           'marginLeft':'5px'},
                                    value='ROQS',
                                    multi=True,
                                    searchable=False
                                )
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
                                                      'name': 'Subjects', 
                                                      'selectable': True}]
                                                ),
                                                #data=[{"Subjects": i} for i in list(df_roqs.index)],
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
                                    outline=True,
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
                                                            [{'id': 'Subjects', 
                                                              'name': 'Subjects', 
                                                              'selectable': True}]
                                                        ),
                                                        #data = [{"Subjects": i} for i in list(names_list)],
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
                # Formation bar plots
                html.Div(
                    id="form-bar-container",
                    className="six columns",
                    children=[
                        build_graph_title("Well count by formations"),
                        dcc.Graph(id="form-by-bar"),
                    ],
                ),
                html.Div(
                    # Selected well productions
                    id="well-production-container",
                    className="six columns",
                    children=[
                        build_graph_title("Individual well annual production"),
                        dcc.Graph(id="production-fig"),
                    ],
                ),
            ],
        ),
    ]
)



'''
# Update bar plot
@app.callback(
    Output("form-by-bar", "figure"),
    [
        Input("well-map", "selectedData"),
        Input("ternary-map", "selectedData"),
        Input("operator-select", "value"),
    ],
)
'''
def update_bar(map_selected_data, tern_selected_data, op_select):
    dff = df[df["op"].isin(op_select)]

    formations = dff["fm_name"].unique().tolist()
    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    processed_data = {}
    if prop_id == "well-map" and prop_type == "selectedData":
        for formation in formations:
            if map_selected_data is None:
                processed_data[formation] = [
                    0
                ]  # [0] is the default value to select current bar
            else:
                processed_data[formation] = get_selection(
                    dff, formation, map_selected_data, 0
                )

    elif prop_id == "ternary-map" and prop_type == "selectedData":

        for formation in formations:
            if tern_selected_data is None:
                processed_data[formation] = [0]
            else:
                processed_data[formation] = get_selection(
                    dff, formation, tern_selected_data, 32
                )

    else:
        for formation in formations:
            processed_data[formation] = [0]

    return generate_formation_bar(dff, processed_data)
'''

# Update ternary map
@app.callback(
    Output("ternary-map", "figure"),
    [
        Input("well-map", "selectedData"),
        Input("form-by-bar", "selectedData"),
        Input("form-by-bar", "clickData"),
        Input("operator-select", "value"),
        Input("ternary-layer-select", "value"),
    ],
    state=[State("ternary-map", "figure")],
)
'''
def update_ternary_map(
    map_selected_data,
    bar_selected_data,
    bar_click_data,
    op_select,
    layer_select,
    curr_fig,
):
    marker_visible = contour_visible = True

    dff = df[df["op"].isin(op_select)]
    formations = dff["fm_name"].unique().tolist()

    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    processed_data = {}

    if prop_id != "ternary-layer-select":
        if prop_id == "well-map" and prop_type == "selectedData":

            for formation in formations:
                if map_selected_data is None:
                    processed_data[formation] = None
                else:
                    processed_data[formation] = get_selection(
                        dff, formation, map_selected_data, 0
                    )

        elif prop_id == "form-by-bar" and prop_type == "selectedData":

            processed_data = get_selection_by_bar(bar_selected_data)

            for formation in formations:
                if bar_selected_data is None:
                    processed_data[formation] = None
                elif formation not in processed_data:
                    processed_data[formation] = []

        elif prop_id == "form-by-bar" and prop_type == "clickData":

            processed_data = get_selection_by_bar(bar_click_data)
            for formation in formations:
                if bar_click_data is None:
                    processed_data[formation] = None
                elif formation not in processed_data:
                    processed_data[formation] = []

        else:

            for formation in formations:
                processed_data[formation] = None

        return generate_ternary_map(
            dff, processed_data, contour_visible, marker_visible
        )

    if prop_id == "ternary-layer-select":
        if curr_fig is not None:
            if "Well Data" not in layer_select:
                marker_visible = "legendonly"
            if "Rock Type" not in layer_select:
                contour_visible = "legendonly"

            for contour_dict in curr_fig["data"][:32]:
                contour_dict["visible"] = contour_visible

            for marker_dict in curr_fig["data"][32:]:
                marker_dict["visible"] = marker_visible
            return curr_fig
        else:
            return curr_fig
'''

# Update well map
@app.callback(
    Output("well-map", "figure"),
    [
        Input("ternary-map", "selectedData"),
        Input("form-by-bar", "selectedData"),
        Input("form-by-bar", "clickData"),
        Input("operator-select", "value"),
        Input("mapbox-view-selector", "value"),
    ],
)
'''
def update_well_map(
    tern_selected_data, bar_selected_data, bar_click_data, op_select, mapbox_view
):
    dff = df[df["op"].isin(op_select)]
    formations = dff["fm_name"].unique().tolist()

    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    processed_data = {}

    if prop_id == "ternary-map":
        for formation in formations:
            if tern_selected_data is None:
                processed_data[formation] = None
            else:
                processed_data[formation] = get_selection(
                    dff, formation, tern_selected_data, 32
                )

    elif prop_id == "form-by-bar":

        bar_data = ""
        if prop_type == "selectedData":
            bar_data = bar_selected_data
        elif prop_type == "clickData":
            bar_data = bar_click_data

        processed_data = get_selection_by_bar(bar_data)

        for formation in formations:
            if bar_data is None:
                processed_data[formation] = None
            elif formation not in processed_data:
                processed_data[formation] = []

    else:
        for formation in formations:
            processed_data[formation] = None

    return generate_well_map(dff, processed_data, mapbox_view)
'''

# Update production plot
@app.callback(
    Output("production-fig", "figure"),
    [
        Input("well-map", "selectedData"),
        Input("ternary-map", "selectedData"),
        Input("form-by-bar", "selectedData"),
        Input("operator-select", "value"),
    ],
)
'''
def update_production(map_select, tern_select, bar_select, op_select):
    dff = df[df["op"].isin(op_select)]

    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    processed_data_init = {}
    processed_data_init["well_id"] = dff["RecordNumber"].tolist()
    processed_data_init["formation"] = dff["fm_name"].tolist()

    if prop_id == "well-map" and prop_type == "selectedData":
        if map_select is not None:
            processed_data = {"well_id": [], "formation": []}
            for point in map_select["points"]:
                processed_data["well_id"].append(point["customdata"])
                processed_data["formation"].append(
                    dff[dff["RecordNumber"] == point["customdata"]]["fm_name"].tolist()[
                        0
                    ]
                )
        else:
            processed_data = processed_data_init

    elif prop_id == "ternary-map" and prop_type == "selectedData":
        if tern_select is not None:
            processed_data = {"well_id": [], "formation": []}
            for point in tern_select["points"]:
                if "customdata" in point:
                    processed_data["well_id"].append(point["customdata"])
                    processed_data["formation"].append(
                        dff[dff["RecordNumber"] == point["customdata"]][
                            "fm_name"
                        ].tolist()[0]
                    )

        else:
            processed_data = processed_data_init

    elif prop_id == "form-by-bar" and prop_type == "selectedData":
        if bar_select is not None:
            processed_data = {"well_id": [], "formation": []}

            # Find all wells according to selected formation category
            for point in bar_select["points"]:
                selected_form = point["x"]
                selected_well = dff[dff["fm_name"] == point["x"]][
                    "RecordNumber"
                ].tolist()
                for well in selected_well:
                    processed_data["well_id"].append(int(well))
                    processed_data["formation"].append(selected_form)

        else:
            processed_data = processed_data_init
    else:
        processed_data = processed_data_init

    return generate_production_plot(processed_data)


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
