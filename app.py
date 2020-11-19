
import os
import pathlib
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from skimage import measure

import warnings
warnings.filterwarnings('ignore') 

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

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
import dash_daq as daq
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate

from dash_extensions import Download
from dash_extensions.snippets import send_data_frame

import plotly.io as pio
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


theme = 'plotly'


# GENERAL DEFINITIONS -------------------------------------------------------------------------

dict_segmentation_functions = {'ROQS': segmfuncs.segm_roqs,
                             'Watershed': segmfuncs.segm_watershed}

dict_parcellation_functions = {'Witelson': parcelfuncs.parc_witelson,
                             'Hofer': parcelfuncs.parc_hofer,
                             'Chao': parcelfuncs.parc_chao,
                             'Cover': parcelfuncs.parc_cover,
                             'Freesurfer': parcelfuncs.parc_freesurfer}

dict_3d_segmentation_functions = {'Watershed3d': segmfuncs.segm_watershed_3d}

scalar_list = ['FA', 'MD', 'RD', 'AD']
colors_list = px.colors.qualitative.Plotly

dict_parcellation_methods = {'Witelson': 'witelson', 'Hofer & Frahm': 'hofer', 'Chao et al':'chao', 'Cover et al': 'cover', 'Freesurfer':'freesurfer'}
dict_segmentation_methods = {'ROQS': 'roqs', 'Watershed': 'watershed'}
dict_3d_segmentation_methods = {'Watershed3d':'watershed3d'}

# DATA IMPORTING -----------------------------------------------------------------------------

# Arg parser
opts = inputfuncs.get_parser().parse_args()

df_categories = pd.DataFrame()
df_numerical = pd.DataFrame()

# Read external data
if opts.ext_data is not None:
    
    external_data = pd.read_excel(external_data_path, dtype={'Subjects':'object'})
    external_data = external_data.set_index('Subjects')

    df_categories = external_data.select_dtypes(include=['object'])
    df_numerical = external_data.select_dtypes(include=np.number)

    col_categories = ['Method'] + list(df_categories.columns)    

# Get indicated directories
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
            dict_folders = dict.fromkeys(directory_dict.keys(), [parent.rsplit('/', 1)[1]])
            group_dict.update(dict_folders)


df_group = pd.DataFrame.from_dict(group_dict, orient='index', columns=["Folder"])
df_categories = pd.concat([df_categories, df_group], axis = 1)

# DATA PROCESSING -----------------------------------------------------------------------------

# Create dataframe for each segmentation method

scalar_statistics_names = ['FA','FA StdDev','MD','MD StdDev','RD','RD StdDev','AD','AD StdDev']
scalar_midline_names = list(range(0,200))

loaded_subjects = []

dict_segmentation_masks = {}
dict_scalar_maps = {}
dict_scalar_statistics = {}
dict_scalar_midlines = {}
dict_error_prob = {}
dict_parcellations_masks = {}
dict_parcellations_statistics = {}
dict_thickness = {}

# Segment and get info

for subject_path in tqdm(path_dict.values()):
    
    for segmentation_method in dict_segmentation_methods.keys():    
        
        if segmentation_method not in dict_scalar_statistics.keys(): 
            dict_segmentation_masks[segmentation_method] = {}
            dict_scalar_maps[segmentation_method] = {}
            dict_scalar_statistics[segmentation_method] = {}
            dict_scalar_midlines[segmentation_method] = {}
            dict_error_prob[segmentation_method] = {}
            dict_parcellations_masks[segmentation_method] = {}
            dict_thickness[segmentation_method] = {}


        # Get data path info
        folderpath = subject_path + 'inCCsight/'
        filename = 'segm_' + dict_segmentation_methods[segmentation_method] + '_data.npy'
        subject_name = subject_path.rsplit('/', 2)[1]

        # Process/Load data
        segmentation_mask, scalar_maps, scalar_statistics, scalar_midlines, error_prob, parcellations_masks = ccprocess.segment(subject_path, 
                                                                                                                               segmentation_method, 
                                                                                                                               dict_segmentation_functions, 
                                                                                                                               dict_parcellation_functions, 
                                                                                                                               opts.basename)        
        # Get thickness
        try:
            thick, _, _ = thickness(segmentation_mask, 200)
        except:
            thick = np.empty(200)
        
        # Assemble dictionaries
        dict_segmentation_masks[segmentation_method][subject_name] = segmentation_mask
        dict_scalar_maps[segmentation_method][subject_name] = scalar_maps
        dict_scalar_statistics[segmentation_method][subject_name] = scalar_statistics
        dict_scalar_midlines[segmentation_method][subject_name] = scalar_midlines
        dict_error_prob[segmentation_method][subject_name] = error_prob
        dict_parcellations_masks[segmentation_method][subject_name] = parcellations_masks
        dict_thickness[segmentation_method][subject_name] = thick
        
        # Save array with subject keys
        loaded_subjects.append(subject_name)
    
    
for segmentation_method in dict_segmentation_methods.keys():

    # Convert to pandas dataframe
    dict_scalar_statistics[segmentation_method] = pd.DataFrame.from_dict(dict_scalar_statistics[segmentation_method], 
                                                                         orient='index', 
                                                                         columns=scalar_statistics_names)
    dict_scalar_midlines[segmentation_method] = pd.DataFrame.from_dict(dict_scalar_midlines[segmentation_method], 
                                                                       orient='index')
    dict_thickness[segmentation_method] = pd.DataFrame.from_dict(dict_thickness[segmentation_method],
                                                                 orient='index')
    dict_error_prob[segmentation_method] = pd.DataFrame.from_dict(dict_error_prob[segmentation_method], columns=['error_prob'], orient='index')
    
    dict_parcellations_statistics[segmentation_method] = inputfuncs.parcellations_dfs_dicts(dict_scalar_maps[segmentation_method], dict_parcellations_masks[segmentation_method], segmentation_method)


# VISUALIZATION -------------------------------------------------------------------------------

app = dash.Dash(__name__, 
               meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=.8, maximum-scale=.8"}],
               external_stylesheets = [dbc.themes.BOOTSTRAP],
               prevent_initial_callbacks=True)
server = app.server
app.config["suppress_callback_exceptions"] = True


# ------------------------------- BUILD FUNCS  -----------------------------------------------

def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Img(src=app.get_asset_url("unicampw.png")),
            html.Img(src=app.get_asset_url("miclab.png")),
            html.Img(src=app.get_asset_url("inccsight.png"), className='logo', style=dict(justifyContent='center')),
            
            #html.H5("Corpus Callosum DTI", className='cctitle'),
            #html.H5("Data Explorer", className='detitle'),
        ],
    )

def build_graph_title(title):
    return html.P(className="graph-title", children=title)

# DataViz ------------------------------------------------------------------------------------

def build_group_segm_boxplot(mode='Method', segmentation_method='ROQS', extra_dims=[]):

    std_colors = pio.templates[theme]['layout']['colorway']
        
    if mode == 'Method':
    
        df = pd.DataFrame()
        scalar_names = ['FA', 'MD', 'RD', 'AD']
        subplots = make_subplots(rows=1, cols=len(scalar_names), subplot_titles=scalar_names)
    
        for i, scalar in enumerate(scalar_names):
            
            for j, segmentation_method in enumerate(dict_segmentation_methods.keys()):
                
                df = dict_scalar_statistics[segmentation_method]
                
                subplots.add_trace(go.Box(y=df[scalar], name=segmentation_method, legendgroup=segmentation_method, marker=dict(color=std_colors[j])), row=1, col=i+1)

                if i == 0:
                    subplots.data[-1].update(name=segmentation_method, legendgroup=segmentation_method)
                else:
                    subplots.data[-1].update(showlegend=False)

    else:

        scalar_names = ['FA', 'MD', 'RD', 'AD'] + extra_dims
        subplots = make_subplots(rows=1, cols=len(scalar_names), subplot_titles=scalar_names)
        
        if len(extra_dims) == 0:
            df = pd.DataFrame()
        else:
            df = df_numerical[extra_dims]
    
        df = pd.concat([df, df_categories[mode], dict_scalar_statistics[segmentation_method]], axis=1)
        df = df.dropna(axis=0)
        
        categories = set(df[mode])
        
        for i, scalar in enumerate(scalar_names):
            
            for j, category in enumerate(categories):
                
                subplots.add_trace(go.Box(y=df[df[mode] == category][scalar], name=category, legendgroup=category, marker=dict(color=std_colors[j])), row=1, col=i+1)

                if i == 0:
                    subplots.data[-1].update(name=category, legendgroup=category)
                else:
                    subplots.data[-1].update(showlegend=False)

    subplots.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h")
    subplots.update_layout(font=dict(family="Open Sans, sans-serif", size=12), margin=dict(r=0, l=0, t=40))
    
    return subplots

def build_parcel_boxplot(scalar='FA', mode='Method', segmentation_method='ROQS', parcellation_method='Witelson'):

    std_colors = pio.templates[theme]['layout']['colorway']
    
    list_regions = ['P1', 'P2', 'P3', 'P4', 'P5']
    subplots = make_subplots(rows=1, cols=5, subplot_titles=list_regions)
    
    if mode == 'Method':
        
        for i, region in enumerate(list_regions):
            
            for j, segmentation_method in enumerate(dict_segmentation_methods.keys()):
                df = dict_parcellations_statistics[segmentation_method][parcellation_method][region][scalar]
                df = df.dropna(axis=0)
                subplots.add_trace(go.Box(y=df, name=segmentation_method, legendgroup=segmentation_method, marker=dict(color=std_colors[j])), row=1, col=i+1)

                if i == 0:
                    subplots.data[-1].update(name=segmentation_method, legendgroup=segmentation_method)
                else:
                    subplots.data[-1].update(showlegend=False)

    else:
    
        categories = list(set(df_categories[mode]))

        df = pd.DataFrame()

        for category in categories:
            df_aux = pd.DataFrame()
            for region in list_regions:
                df_aux = pd.concat([df_aux, dict_parcellations_statistics[segmentation_method][parcellation_method][region][scalar][df_categories[mode] == category]], axis=1)
            df_aux[mode] = category
            df = pd.concat([df, df_aux], axis=0)
        
        df = df.dropna(axis=0)
        names = list_regions + [mode]
        df.columns = names

        categories = set(df[mode])
        
        for i, region in enumerate(list_regions):
            
            for j, category in enumerate(categories):
                
                subplots.add_trace(go.Box(y=df[df[mode] == category][region], name=category, legendgroup=category, marker=dict(color=std_colors[j])), row=1, col=i+1)

                if i == 0:
                    subplots.data[-1].update(name=category, legendgroup=category)
                else:
                    subplots.data[-1].update(showlegend=False)

    subplots.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h")
    subplots.update_layout(font=dict(family="Open Sans, sans-serif", size=12), margin=dict(r=0, l=0, t=40))
    
    return subplots

def build_segm_scatterplot(mode='Method', segmentation_method = 'ROQS', scalar_x = 'FA', scalar_y = 'MD'):

    df = pd.DataFrame()

    if mode == 'Method':
        df = pd.DataFrame()
        for segmentation_method in dict_segmentation_methods.keys():
            df_aux = dict_scalar_statistics[segmentation_method]
            df_aux['Method'] = segmentation_method 
            df = pd.concat([df, df_aux], axis=0)

    else:
        df = pd.concat([df_categories[mode], dict_scalar_statistics[segmentation_method]], axis=1)
        df = pd.concat([df, df_numerical], axis=1)        
        
    df = df.dropna(axis=0)
    fig = px.scatter(df, 
                    x=scalar_x, 
                    y=scalar_y, 
                    color=mode, 
                    marginal_y="violin", 
                    marginal_x="histogram",
                    hover_name=df.index)

    fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)',legend_orientation="h")
    fig.update_layout(font=dict(family="Open Sans, sans-serif", size=12))
    
    return fig

def build_segm_scattermatrix(mode='Method', segmentation_method = 'ROQS', extra_dims = []):

    dimensions = ['FA','MD','RD','AD']
    df = pd.DataFrame()

    if mode == 'Method':
        for segmentation_method in dict_segmentation_methods.keys():
            df_aux = dict_scalar_statistics[segmentation_method]
            df_aux['Method'] = segmentation_method 
            df = pd.concat([df, df_aux], axis=0)
    
    else:
        df = pd.concat([df_categories[mode], dict_scalar_statistics[segmentation_method]], axis=1)
    
        if len(extra_dims) > 0:
            dimensions = ['FA','MD','RD','AD'] + extra_dims
            df = pd.concat([df, df_numerical], axis=1)

    df = df.dropna(axis=0)
    fig = px.scatter_matrix(df, 
                            dimensions=dimensions,
                            color=mode,
                            hover_name=df.index)

    fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)',legend_orientation="h")
    fig.update_layout(font=dict(family="Open Sans, sans-serif", size=12), margin=dict(r=0, l=0))

    return fig

def build_midline_plot(scalar='FA', mode='Method', segmentation_method='ROQS'):

    df = pd.DataFrame()
    
    if mode == 'Method':
        for segmentation_method in dict_segmentation_methods.keys():
            
            if scalar in ['FA', 'MD', 'AD', 'RD']:
                df_aux = pd.DataFrame.from_dict(dict(dict_scalar_midlines[segmentation_method][scalar]), orient='index', columns=scalar_midline_names)
            elif scalar == 'Thickness':
                df_aux = dict_thickness[segmentation_method]
            
            df_aux['Method'] = segmentation_method    
            df = pd.concat([df, df_aux], axis=0)
        df = df.dropna(axis=0)
            
        df_grouped = df.groupby('Method').mean().transpose()
        df_melt = pd.melt(df_grouped.reset_index(), id_vars='index', value_vars=set(df[mode]))
    
    else:
        if scalar in ['FA', 'MD', 'AD', 'RD']:
            df_aux = pd.DataFrame.from_dict(dict(dict_scalar_midlines[segmentation_method][scalar]), orient='index', columns=scalar_midline_names)
        elif scalar == 'Thickness':
            df_aux = dict_thickness[segmentation_method]
            
        df = pd.concat([df_categories[mode], df_aux], axis=1)
        df = df.dropna(axis=0)

        df_grouped = df.groupby(mode).mean().transpose()
        df_melt = pd.melt(df_grouped.reset_index(), id_vars='index', value_vars=set(df[mode]))
        
    fig = px.line(df_melt, x='index', y='value', color=mode)

    fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h")    
    fig.update_layout(font=dict(family="Open Sans, sans-serif", size=12))
    
    fig.update_layout(xaxis_title='Points along CC body', yaxis_title=scalar, legend_title=mode)
    
    return fig

def build_bubble_grouped(mode='Method', segmentation_method='ROQS', scalar='Thickness', size=True):
    
    def build_bubble_plot(scalar='FA', segmentation_method='Watershed', size = True, category_index = None):

        df_pts = pd.read_pickle('./assets/bubble_plot_xy.pkl')
        segm_contour = np.load('./assets/bubble_plot_contour.npy')

        if scalar == 'Thickness':
            df_aux = dict_thickness[segmentation_method][list(np.linspace(0,195,40))+[199]]    
        else:
            df_aux = pd.DataFrame.from_dict(dict(dict_scalar_midlines[segmentation_method][scalar]), orient='index', columns=scalar_midline_names)[list(np.linspace(0,195,40))+[199]]

        if category_index is not None:
            df_aux = df_aux.loc[category_index]
        df_pts[scalar] = df_aux.mean().reset_index()[0]

        if size == True:
            fig = px.scatter(df_pts, x="x", y="y", color = scalar, size = df_pts['Thickness'])
            fig.update_traces(marker=dict(sizeref = 2. * max(df_pts[scalar]) / (45 ** 2)))
        else:
            fig = px.scatter(df_pts, x="x", y="y", color=scalar)
            fig.update_traces(marker=dict(size=25))

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h")    
        fig.update_layout(font=dict(family="Open Sans, sans-serif", size=12))

        fig.add_trace(go.Contour(z=segm_contour, contours=dict(start=0, end=70, size=70, coloring='none'),
                                 showscale=False, line_width=3, line=dict(color='rgba(0,0,0,.5)', dash='dot')))

        return fig
    
    if mode == 'Method':
        
        n_cats = len(dict_segmentation_methods.keys())
        fig = make_subplots(rows=n_cats, cols=1, vertical_spacing=0, shared_xaxes=True, xaxis_title='Points along CC body')
        
        for i, segmentation_method in enumerate(dict_segmentation_methods):
            fig.add_trace(build_bubble_plot(scalar=scalar, segmentation_method=segmentation_method, size=size)['data'][0], row=i+1, col=1)
            fig.update_yaxes(title_text="{} for {} method".format(scalar, segmentation_method), row=i+1, col=1)
        
    else:
        
        df = df_categories[mode]
        df = df.dropna(axis=0)
        n_cats = len(set(df))
        fig = make_subplots(rows=n_cats, cols=1)
        
        for i, category in enumerate(set(df)):
            category_index = df_categories.loc[df_categories[mode] == category].index
            fig.add_trace(build_bubble_plot(scalar=scalar, segmentation_method=segmentation_method, size=size, category_index=category_index)['data'][0], row=i+1, col=1)
            fig.update_yaxes(title_text="{} for {} category".format(scalar, category), row=i+1, col=1)
    
    fig.update_xaxes(title_text="Points along CC body", row=n_cats, col=1)
    fig.update_layout(height=300*n_cats, width=800) 
    fig.update_layout(font=dict(family="Open Sans, sans-serif", size=12))
    return fig

def build_bubble_grouped_pvalue(scalar='Thickness', mode='Method', segmentation_method='ROQS', threshold=0.05):
    
    def build_bubble_pvalue(pvalue, threshold=0.05, size=False, gray=False):

        df_pts = pd.read_pickle('./assets/bubble_plot_xy.pkl')
        segm_contour = np.load('./assets/bubble_plot_contour.npy')
        marker_color = 'rgba(100,100,100,0.5)'

        df_pts['p-value'] = pvalue

        if gray:
            fig = px.scatter(df_pts, x="x", y="y", hover_data=['p-value'])
            fig.update_traces(marker=(dict(color=marker_color)))
        else:
            fig = px.scatter(df_pts.loc[df_pts['p-value'] < threshold], x="x", y="y", color = 'p-value')
        fig.update_traces(marker=dict(size=25))

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h")    
        fig.update_layout(font=dict(family="Open Sans, sans-serif", size=12))

        return fig
    
    if mode == 'Method':
        categories = list(itertools.combinations(dict_segmentation_methods.keys(), 2))
        n_cats = len(categories)
        
        fig = make_subplots(rows=n_cats, cols=1)
        
        for i, category in enumerate(categories):
            
            if scalar == 'Thickness':
                pvalue = ttest_ind(dict_thickness[category[0]], dict_thickness[category[1]]).pvalue
            else:
                pvalue = ttest_ind(pd.DataFrame.from_dict(dict(dict_scalar_midlines[category[0]][scalar]), orient='index', columns=scalar_midline_names),
                                   pd.DataFrame.from_dict(dict(dict_scalar_midlines[category[1]][scalar]), orient='index', columns=scalar_midline_names)).pvalue

            pvalue = np.take(pvalue, list(np.linspace(0,195,40)))

            new_gray_fig = build_bubble_pvalue(pvalue, gray=True)
            
            new_fig = build_bubble_pvalue(pvalue, threshold)['data']
            
            if len(new_fig) > 0:
                fig.add_trace(new_gray_fig['data'][0], row=i+1, col=1)
                fig.add_trace(new_fig[0], row=i+1, col=1)
            else:
                fig.add_trace(new_gray_fig['data'][0], row=i+1, col=1)
                
            fig.update_yaxes(title_text="{} x {}".format(category[0], category[1]), row=i+1, col=1)
            
    else:
        
        df = df_categories[mode]
        df = df.dropna(axis=0)
        categories = list(itertools.combinations(set(df), 2))
        n_cats = len(categories)
        
        fig = make_subplots(rows=n_cats, cols=1)
        
        for i, category in enumerate(categories):
            
            if scalar == 'Thickness':
                pvalue = ttest_ind(dict_thickness[segmentation_method].loc[df_categories[mode] == category[0]], 
                                   dict_thickness[segmentation_method].loc[df_categories[mode] == category[1]]).pvalue
            else:
                pvalue = ttest_ind(pd.DataFrame.from_dict(dict(dict_scalar_midlines[segmentation_method][scalar]), orient='index', columns=scalar_midline_names).loc[df_categories[mode] == category[0]],
                                   pd.DataFrame.from_dict(dict(dict_scalar_midlines[segmentation_method][scalar]), orient='index', columns=scalar_midline_names).loc[df_categories[mode] == category[1]]).pvalue
    
            pvalue = np.take(pvalue, list(np.linspace(0,195,40)))

            new_gray_fig = build_bubble_pvalue(pvalue, gray=True)
            
            new_fig = build_bubble_pvalue(pvalue, threshold)['data']
            
            if len(new_fig) > 0:
                fig.add_trace(new_gray_fig['data'][0], row=i+1, col=1)
                fig.add_trace(new_fig[0], row=i+1, col=1)
            else:
                fig.add_trace(new_gray_fig['data'][0], row=i+1, col=1)
                
            fig.update_yaxes(title_text="{} x {}".format(category[0], category[1]), row=i+1, col=1)
           
    fig.update_layout(height=300*n_cats, width=800) 
    fig.update_xaxes(title_text="Points along CC body", row=n_cats, col=1)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h")    
    fig.update_layout(font=dict(family="Open Sans, sans-serif", size=12))

    return fig

# -----------------------------------------------------------------------------------------

def build_scalar_compare_dropdowns():

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

def build_parcel_boxplot_dropdowns():

    options_scalars = [{'label': scalar, 'value': scalar} for scalar in scalar_list]
    options_parcel_method = [{'label': parc, 'value': parc} for parc in dict_parcellations_statistics[segmentation_method].keys()]

    layout = html.Div([
                html.Div([
                    html.H6('Parc. Method:', className='table-options-title', style={'padding':'0px 10px 0px 10px'}),
                    dcc.Dropdown(id='dropdown-parcel-boxplot-left',
                                 options=options_parcel_method,
                                 multi=False,
                                 value=list(dict_parcellation_methods.keys())[0],
                                 style={'width':'150px'}),
                    html.H6('Scalar:', className='table-options-title', style={'padding':'0px 10px 0px 30px'}),
                    dcc.Dropdown(id='dropdown-parcel-scalars-right',
                                 options=options_scalars,
                                 multi=False,
                                 value='FA',
                                 style={'width':'120px'})
                ], className='row', style=dict(display='flex', justifyContent='left', verticalAlign='center')),
            ]
        )
    return layout    

def built_scalar_dropdown():

    options = [{'label': scalar, 'value': scalar} for scalar in scalar_list]

    layout = html.Div([
                html.Div([
                    dcc.Dropdown(id='dropdown-scalars',
                                 options=options,
                                 multi=False,
                                 value='FA',
                                 style={'float': 'right', 'width':'150px'}),
                ], className='row', style={'justifyContent':'center'}),
            ]
        )
    return layout

def build_quality_collapse():

    layout = dbc.Card([
                
                build_graph_title("Quality evaluation"),
                
                html.Div([
                    html.H5("Threshold:"), 
                    dcc.Dropdown(id='dropdown-quality-threshold',
                                 options= [{'label': num/100, 'value': num/100} for num in np.arange(95, 5, -5)],
                                 multi=False,
                                 value='0.7',
                                 style={'float': 'right', 'width':'150px', 'margin-left':'10px'})
                    ], className='row', 
                    style=dict(margin="1rem 0 0 3rem", display="flex", alignItems="center")),
                
                html.Div(children=build_quality_images(), 
                    className="row", 
                    id='photo-container',
                    style=dict(margin="2rem 1rem 2rem 3rem", height=400, width="95%", overflowY="auto"))
            
            ], style={'margin':'20px', 'backgroundColor':'#FAFAFA', 'border-radius':'20px'})

    return layout

def build_quality_images(threshold=0.7):
    
    children = []

    for segmentation_method in dict_segmentation_functions.keys():
        
        # Get error probs
        df = dict_error_prob[segmentation_method]
        df.sort_values(by=['error_prob'])
        
        # Order by error probs
        index_label = df.query('error_prob > '+str(threshold)).index.tolist()

        # Retrieve images and segmentation
        for subject_id in index_label:

            folderpath = path_dict[subject_id] + 'inCCsight/'
            filepath = folderpath + 'segm_' + dict_segmentation_methods[segmentation_method] + '_data.npy'

            children.append(html.Div([
                                html.Div([
                                    html.H6("Subject: {} - Method: {}".format(subject_id, segmentation_method)),
                                    dbc.Button("Remove", color='dark', size='lg', id={'type': 'remove_button', 'index': 'btn_{}_{}'.format(segmentation_method, subject_id)}),
                                    ], className='row', style=dict(width='100%', display='flex', alignItems='center', justifyContent='space-between')),
                                dcc.Graph(figure=build_fissure_image(filepath))
                            ], className = 'three columns'))
    return children

def build_3d_visualization(subject_id):

    folderpath = path_dict[subject_id] + 'inCCsight/'
    filepath = folderpath + 'segm_watershed3d.npy'

    if os.path.exists(filepath):
       
        _, segmentation3d, wFA_v, _ = np.load(filepath, allow_pickle=True)

        verts, faces, normals, values = measure.marching_cubes_lewiner(wFA_v, 0)

        tri_FA = ff.create_trisurf(x=verts[:,0], y=verts[:,2], z=verts[:,1]*-1+70,
                                simplices=faces,
                                colormap=[(1,0,0), (1,0,0)],
                                aspectratio=dict(x=1, y=1, z=.66),
                                plot_edges = False,
                                show_colorbar = False)

        tri_FA['data'][0].update(opacity=0.2)

        verts, faces, normals, values = measure.marching_cubes_lewiner(segmentation3d, 0)

        tri_CC = ff.create_trisurf(x=verts[:,0], y=verts[:,2], z=verts[:,1]*-1+70,
                                simplices=faces,
                                colormap=[(0,0,1), (0,0,1)],
                                aspectratio=dict(x=1, y=1, z=.66),
                                plot_edges = False,
                                show_colorbar = False)

        tri_CC['data'][0].update(opacity=0.1)

        fig = go.Figure(tri_FA)
        fig.add_trace(tri_CC.data[0])
        fig.update_layout(title="3D Visualization")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        return fig

    else:
        return []

def build_subject_collapse(segmentation_method='ROQS', scalar_map='wFA', subject_id = list(path_dict.keys())[0]):

    folderpath = path_dict[subject_id] + 'inCCsight/'
    filepath = folderpath + 'segm_' + name_dict[segmentation_method] + '_data.npy'

    layout = dbc.Card([
                build_graph_title("Subject " + subject_id),
                html.Div([                
    
                    # Segmentation image
                    html.Div([
                        dcc.Graph(figure=build_3d_visualization(subject_id))
                    ], className = 'three columns'),

                ], className='row', style={'justifyContent':'center'}),

            ], style={'margin':'20px', 'backgroundColor':'#FAFAFA', 'border-radius':'20px'})
    return layout

def build_fissure_image(filepath, scalar = 'wFA'):

    scalar_maps_list = ['wFA','FA','MD','RD','AD']
    segmentation, scalar_maps = np.load(filepath, allow_pickle=True)[:2]
    img = scalar_maps[scalar_maps_list.index(scalar)]

    contours = measure.find_contours(segmentation, 0.1)
    contour = sorted(contours, key=lambda x: len(x))[-1]

    fig = px.imshow(img, color_continuous_scale='gray', aspect='auto')
    fig.add_trace(go.Scatter(x=contour[:, 1], y=contour[:, 0]))
    fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h", coloraxis_showscale=False)
    fig.update_layout(margin = dict(l=0, r=0,t=0,b=0))

    return fig 

# DataTable functions ---------------------------------------------------------------------

def extend_colorscale(colormap, factor = 5):
    
    from plotly.colors import n_colors
    
    new_colormap = []
    for i in range(len(colormap)-1):
        new_colormap += n_colors(eval(colormap[i][3:]), eval(colormap[i+1][3:]), factor)
    
    for i, color in enumerate(new_colormap):
        new_colormap[i] = 'rgb' + str(color)
        
    return new_colormap
   
def color_table(df, use_values_limits = True, mean = None, stdev = None):

    if 'id' in df:
        numeric_columns = df.select_dtypes('number').drop(['index'], axis=1)
    else:
        numeric_columns = df.select_dtypes('number')

    colormap = px.colors.diverging.RdBu[::-1][2:-2]
    colormap = extend_colorscale(colormap, 4)
    
    styles = []

    for col in numeric_columns:

        values = df[col]

        if use_values_limits is True:
            mean = np.mean(values)
            min_value = np.min(values)
            max_value = np.max(values)
        else:
            min_value = mean - stdev
            max_value = mean + stdev
            
        limits = np.linspace(min_value, max_value, len(colormap))
            
        for value in values:
            idx = (np.abs(limits - value)).argmin()
            
            styles.append({
                'if': {
                    'filter_query': '{{{col}}} = {value}'.format(
                            col = col, value = value
                        ),
                    'column_id': col
                    },
                    'backgroundColor': colormap[idx]
                })

            styles.append({
                'if': {
                    'filter_query': '{{{col}}} = {value}'.format(col = col, value = value),
                    'state': 'selected',
                    'column_id': col,
                    },
                    'backgroundColor': colormap[idx]
                })

            styles.append({
                'if': {
                    'filter_query': '{{{col}}} = {value}'.format(col = col, value = value),
                    'state': 'selected',
                    'column_id': 'index',
                    },
                    'backgroundColor': 'rgb(228, 228, 255)'
                })

    return styles

def stripped_rows():

    style_data_conditional = []

    style_data_conditional.append({'if': {'row_index': 'odd'},
                                      'backgroundColor': 'rgb(248, 248, 248)'})
    style_data_conditional.append({'if': {'row_index': 'odd', 'state': 'selected'},
                                   'backgroundColor': 'rgb(228, 228, 228)'})
    style_data_conditional.append({'if': {'row_index': 'even'},
                                  'backgroundColor': 'rgb(255, 255, 255)'})
    style_data_conditional.append({'if': {'row_index': 'even', 'state': 'selected'},
                                   'backgroundColor': 'rgb(235, 235, 235)'})
    style_data_conditional.append({'if': {'state': 'selected'},
                                   "border": "3px solid blue"})
    return style_data_conditional

def build_segm_table(mode = 'Method', segmentation_method = 'ROQS', show_stdev = False, color = False):

    list_scalars = ['FA','RD','AD','MD']

    if mode == 'subjects':
        df = dict_scalar_statistics[segmentation_method].reset_index()
        df = df.rename(columns={"index": "Subject"})
        names = ['Subject'] + list_scalars
    
    elif mode == 'Method':
        df = pd.DataFrame()
        for segmentation_method in dict_segmentation_functions.keys():
            df_aux = dict_scalar_statistics[segmentation_method].reset_index()
            df_aux['Method'] = segmentation_method
            df = pd.concat([df, df_aux], axis=0)
        df = df.groupby('Method').mean().reset_index()
        names = ['Method'] + list_scalars

    else:
        df = dict_scalar_statistics[segmentation_method]
        df = pd.concat([df_categories[mode], df], axis=1)
        df = df.groupby(mode).mean().reset_index()
        names = [mode] + list_scalars
    
    df = df.round(6)

    if show_stdev is False:
        columns=[{"name": i, "id": i, "selectable": True} for i in names]
        data=df[names]
    else:
        columns=[{"name": i, "id": i} for i in df.columns[:-1]]
        data=df

    layout = dash_table.DataTable(
        id = 'segm_table',
        columns = columns,
        data = data.to_dict('records'),
        
        page_action = 'none',
        fixed_rows = {'headers': True},
        style_table={
            'maxHeight': '300px', 
            'overflowY': 'auto'},
        style_header = {
            'fontWeight': 'bold',
        },
        style_cell = {
            'font_family': 'Open Sans',
            'font_size': '18px',
            'text_align': 'center'
        },
        style_as_list_view = True,
        export_format='xlsx',
        export_headers='display',
        style_data_conditional = stripped_rows(),
    )

    return layout

def build_parcel_table(mode = 'Method', segmentation_method = 'ROQS', parcellation_method = 'Witelson', scalar = 'FA', color = False):
    
    list_regions = ['P1', 'P2', 'P3', 'P4', 'P5']

    if mode == 'subjects':
        df = pd.DataFrame()
        for region in list_regions:
            df = pd.concat([df, dict_parcellations_statistics[segmentation_method][parcellation_method][region][scalar]], axis=1)
        df.columns = list_regions
        df = df.reset_index()
        df = df.rename(columns={"index": "Subject"})
        names = ['Subject'] + list_regions
    
    elif mode == 'Method':
        
        df = pd.DataFrame()
        for segmentation_method in dict_segmentation_methods.keys():
            df_aux = pd.DataFrame()
            for region in list_regions:
                df_aux = pd.concat([df_aux, dict_parcellations_statistics[segmentation_method][parcellation_method][region][scalar]], axis=1)
            df_aux['Method'] = segmentation_method
            df = pd.concat([df, df_aux], axis=0)
        df = df.groupby('Method').mean()
        df.columns = list_regions
        names = ['Method'] + list_regions

    else:

        categories = list(set(df_categories[mode]))

        df = pd.DataFrame()
        for category in categories:
            df_aux = pd.DataFrame()
            for region in list_regions:
                df_aux = pd.concat([df_aux, dict_parcellations_statistics[segmentation_method][parcellation_method][region][scalar][df_categories[mode] == category]], axis=1)
            df_aux[mode] = category
            df = pd.concat([df, df_aux], axis=0)
        df = df.groupby(mode).mean()
        df.columns = list_regions
        names = [mode] + list_regions            


    layout = dash_table.DataTable(
        id = 'parcel_table',
        columns = [{"name": name, "id": name} for name in names],
        data = df.round(6).reset_index().to_dict('records'),

        page_action = 'none',
        fixed_rows = {'headers': True},
        style_table={
            'maxHeight': '300px', 
            'overflowY': 'auto'},
        style_header = {
            'fontWeight': 'bold',
        },
        style_cell = {
            'font_family': 'Open Sans',
            'font_size': '18px',
            'text_align': 'center'
        },
        style_as_list_view = True,
        export_format='xlsx',
        export_headers='display',
        style_data_conditional=stripped_rows()
        
    )

    return layout

def export_parcel_table(segmentation_method = 'ROQS', parcellation_method = 'Witelson'):

    list_regions = ['P1', 'P2', 'P3', 'P4', 'P5']

    df = pd.concat(dict_parcellations_statistics[segmentation_method][parcellation_method].values(), 
                   axis = 1, 
                   keys = dict_parcellations_statistics[segmentation_method][parcellation_method].keys())

    filename = "inCCsight_parceldata_{}_{}.xlsx".format(segmentation_method, parcellation_method)

    return send_data_frame(df.to_excel, filename)

# ---------------------------------- LAYOUT -----------------------------------------------
app.layout = html.Div(
    children=[
        html.Div(
            id="top-row",
            children=[
            # Top row left
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
                                html.Div(className='row', style=dict(display='flex', justifyContent='center', verticalAlign='center'),
                                    children=[
                                        html.H5("Category:", style=dict(color='white', padding='0 10px 0 54px')),
                                        dcc.Dropdown(id='dropdown_category',
                                            options=[{'label': method, 'value': method} for method in ['Method']+list(set(df_categories.columns))],
                                            multi=False,
                                            value='Method',
                                            style={'width':'150px'})
                                    ],
                                    id="div_category",
                                ),  
                                
                                html.Div(className='row', style=dict(display='flex', justifyContent='center', verticalAlign='center'),
                                    children=[
                                       html.H5("Segm. Method:", style=dict(color='white', padding='0 10px 0 0')),
                                       dcc.Dropdown(id='dropdown_segm_methods',
                                             options=[{'label': method, 'value': method} for method in dict_segmentation_functions.keys()],
                                             multi=False,
                                             value=list(dict_segmentation_functions.keys())[0],
                                             style={'width':'150px'})
                                    ],
                                    id="div_select_segm_method",
                                ),

                                html.Div(className='row', id='quality-button-container', 
                                    children=[
                                       dbc.Button(
                                            ["Check quality  ", dbc.Badge("0", color="warning", className="ml-1", id='quality-count')],
                                            size='lg',
                                            color="light",
                                            id='quality-button'
                                        )
                                ]),
                            
                            ],
                        )
                    ],
                ),

            # Top row right
                html.Div(
                    className="row",
                    id="top-row-graphs",
                    children=[

                        html.Div([

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
                            ], className = "four columns"),

                            # Create buttons
                            html.Div([
                                html.Button('>>', 
                                    id='group-button',
                                ),                            
                            ], id = "button-container", className = "two columns"),

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
                            ], className = "four columns"),

                        ], className="one-half column", id="left-panel"),


                        html.Div([

                            html.Div(
                                id='summary-card-container',
                                className='five columns',
                                children=[
                                dbc.Card(
                                    id = "summary-card",
                                    children = [
                                        html.H1(str(len(set(list(path_dict.keys())))), className='numcard'),
                                        html.H5("Subjects", className='titlecard'),
                                        html.H1(str(len(set(np.hstack(list(group_dict.values()))))), className='numcard', style={'margin-top': '10px'}),
                                        html.H5("Groups", className='titlecard'),
                                    ],
                                ),
                            ]),

                        ], className="one-half column", id="right-panel"),
                    ],
                ),
            ],
        ),
        
        dbc.Collapse(
            dbc.Card(dbc.CardBody(build_quality_collapse())),
            id="quality-collapse",
        ),

        dbc.Collapse(
            dbc.Card(dbc.CardBody("very cool thingy")),
            id="subject-collapse",
        ),

        html.Div(
            className = 'row',
            id = 'tables-row',
            children = [

                html.Div(
                    id = 'segm_table_super_container',
                    className = 'six columns',
                    children = [
                        build_graph_title("Segmentation data"),
                        html.Div(
                            id = 'segm_table_container',
                            children = [build_segm_table(show_stdev = False, color = True)]),
                        html.Div(
                            id = 'segm_table_options',
                            className = 'table-options',
                            children = [

                                html.H6('Mode:', className='table-options-title'),

                                dcc.Dropdown(
                                    id='segm_table_dropdown_mode',
                                    className = 'options-dropdown',
                                    options=[{'label': 'Overall', 'value': 'overall'},
                                              {'label': 'Subjects', 'value': 'subjects'}],
                                    multi=False,
                                    value='overall'),

                                html.H6('Std.Dev.:', className='table-options-title'),

                                dcc.Dropdown(
                                    id = 'segm_table_dropdown_stdev',
                                    className = 'options-dropdown',
                                    options = [
                                        {'label': 'Show', 'value': True},
                                        {'label': 'Hide', 'value': False},
                                    ],
                                    value = False,
                                ),

                            ],
                        ),
                    ],
                ),
                
                html.Div(
                    id = 'parcel_table_super_container',
                    className = 'six columns',
                    children = [
                        build_graph_title("Parcellation data"),
                        
                        html.Div(
                            id = 'parcel_table_container',
                            children = [build_parcel_table(
                                            segmentation_method = 'ROQS', 
                                            parcellation_method = 'Witelson', 
                                            scalar = 'FA')]),
                        
                        html.Div(
                            id = 'parcel_table_options',
                            className = 'table-options',
                            children = [
                                
                                html.H6('Mode:', className='table-options-title'),

                                dcc.Dropdown(
                                    id='parcel_table_dropdown_mode',
                                    className = 'options-dropdown',
                                    options=[{'label': 'Overall', 'value': 'overall'},
                                             {'label': 'Subjects', 'value': 'subjects'}],
                                    multi=False,
                                    value='overall',
                                ),
                                
                                html.H6('Parcel. method:', className='table-options-title'),

                                dcc.Dropdown(
                                    id = 'parcel_table_dropdown_method',
                                    className = 'options-dropdown',
                                    options = [
                                        {'label': 'Witelson', 'value': 'Witelson'},
                                        {'label': 'Hofer & Frahm', 'value': 'Hofer'},
                                        {'label': 'Chao et al.', 'value': 'Chao'},
                                        {'label': 'Cover et al.', 'value': 'Cover'},
                                        {'label': 'Freesurfer', 'value': 'Freesurfer'},
                                    ],
                                    value = 'Witelson',
                                ),

                                html.H6('Scalar:', className='table-options-title'),

                                dcc.Dropdown(
                                    id = 'parcel_table_dropdown_scalar',
                                    className = 'options-dropdown',
                                    options = [
                                        {'label': 'FA', 'value': 'FA'},
                                        {'label': 'MD', 'value': 'MD'},
                                        {'label': 'RD', 'value': 'RD'},
                                        {'label': 'AD', 'value': 'AD'},
                                    ],
                                    value = 'FA'
                                ),                

                                html.Button(
                                    id='parcel_download_all_btn',
                                    children='Download all data', 
                                ),

                                Download(id="parcel_download_all"),

                            ],
                        ),
                    ],
                )
                

            ]
        ),

        html.Div(
            className="row",
            id="bottom-row",
            children=[

                html.Div(
                        id="boxplot-container1",
                        className="eight columns",
                        children=[
                            build_graph_title("Segmentation BoxPlots"),
                            dcc.Graph(id="segm_boxplots", figure=build_group_segm_boxplot()),
                        ],
                    ),

                
                # Midline plots
                html.Div(
                    id="midline-container",
                    className="four columns",
                    children=[
                        build_graph_title("Midline Plots"),
                        dcc.Graph(id="midline_graph", figure=build_midline_plot()),
                        built_scalar_dropdown(),
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
                    id="scattermatrix-container",
                    className="eight columns",
                    children=[
                        build_graph_title("Scatter Matrix"),
                        dcc.Graph(id="scatter_matrix", figure=build_segm_scattermatrix()),
                    ],
                ),
                # Selected well productions
                html.Div(
                    id="scatterplot-container",
                    className="four columns",
                    children=[
                        build_graph_title("Scatter Plot"),
                        dcc.Graph(id="scatter_plot", figure=build_segm_scatterplot()),
                        build_scalar_compare_dropdowns(),
                    ],
                ),

            ],
        ),

        html.Div(
            className="row",
            id="third-bottom-row",
            children=[

                html.Div(
                        id="boxplot-container2",
                        className="eight columns",
                        children=[
                            build_graph_title("Parcellation BoxPlots"),
                            dcc.Graph(id="parc_boxplots", figure=build_parcel_boxplot()),
                            build_parcel_boxplot_dropdowns(),
                        ],
                    ),

                ],
            ),        
    ],
)

# --------------------------------- CALLBACKS ---------------------------------------------

# Enable/Disable segm method dropdown
@app.callback(
    Output("dropdown_segm_methods","disabled"),
    [Input("dropdown_category","value")])
def update_dropdown_disabled(mode):
    if mode == 'Method':
        return True
    else:
        return False

# Update segm box-plots
@app.callback(
    Output("segm_boxplots", "figure"),
    [Input("dropdown_segm_methods","value"),
     Input("dropdown_category","value")])
def update_segm_boxplots(segm_method, mode):
    return build_group_segm_boxplot(mode=mode, segmentation_method=segm_method)

# Update segm box-plots
@app.callback(
    Output("parc_boxplots", "figure"),
    [Input("dropdown_segm_methods","value"),
     Input("dropdown_category","value"),
     Input("dropdown-parcel-boxplot-left","value"),
     Input("dropdown-parcel-scalars-right","value")])
def update_segm_boxplots(segm_method, mode, parc_method, scalar):
    return build_parcel_boxplot(scalar=scalar, mode=mode, segmentation_method=segm_method, parcellation_method=parc_method)

# Update midline plot
@app.callback(
    Output("midline_graph", "figure"),
    [Input("dropdown_segm_methods","value"),
     Input("dropdown_category","value"),
     Input("dropdown-scalars","value")])
def update_midlineplot(segm_method, mode, scalar):
    return build_midline_plot(mode=mode, segmentation_method=segm_method, scalar=scalar)

# Update scatter matrix
@app.callback(
    Output("scatter_matrix", "figure"),
    [Input("dropdown_segm_methods", "value"),
     Input("dropdown_category","value")])
def update_scattermatrix(segm_method, mode):
    return build_segm_scattermatrix(mode=mode, segmentation_method=segm_method)

# Update scatter plot
@app.callback(
    Output("scatter_plot","figure"),
    [Input("dropdown_segm_methods","value"),
     Input("dropdown_category","value"),
     Input("dropdown-scalars-right","value"),
     Input("dropdown-scalars-left","value")])
def update_scatterplot(segm_method, mode, scalar_x, scalar_y):
    return build_segm_scatterplot(mode=mode, segmentation_method=segm_method, scalar_x=scalar_x, scalar_y=scalar_y)

# Update number of quality warnings
@app.callback(
    Output("quality-count", 'children'),
    [Input("photo-container", 'children')])
def update_quality_counter(quality_photos):
    return len(quality_photos)

# Open quality collapse
@app.callback(
    Output("quality-collapse","is_open"),
    [Input("quality-button","n_clicks")])
def open_quality_collapse(n_clicks):
    if n_clicks == None or n_clicks%2 == 0:
        return False
    else: 
        return True

# Update quality threshold
@app.callback(
    Output("photo-container","children"),
    [Input("dropdown-quality-threshold","value")])
def update_quality_threshold(threshold):
    return build_quality_images(threshold)

# Open subject collapse
@app.callback(
    [Output("subject-collapse", "is_open"),
     Output("subject-collapse", "children")],
    [Input("table-subjects","selected_cells")])
def open_subject_collapse(selected_cells):
    if len(selected_cells) == 1:
        subject_list = list(path_dict.keys())
        subject_id = subject_list[selected_cells[0]['row']]
        return True, [build_subject_collapse(subject_id = subject_id)]
    else:
        return False, []

# Remove subject in quality view
@app.callback(
    Output('subject_list', 'children'),
    [Input({'type': 'remove_button', 'id': ALL}, 'n_clicks')])
def remove_subject_from_quality(value):
    print(value)

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

# Change segm table mode
@app.callback(
    Output("segm_table_container", "children"),
    [Input("segm_table_dropdown_mode", "value"),
     Input("dropdown_segm_methods", "value"),
     Input("dropdown_category","value"),
     Input("segm_table_dropdown_stdev", "value")])
def change_segm_table_mode(table_mode, segmentation_method, mode, show_stdev):
    if table_mode == 'subjects':
        mode = 'subjects'
    return [build_segm_table(mode = mode, segmentation_method = segmentation_method, show_stdev = show_stdev, color = False)]

# Change parcel table mode
@app.callback(
    Output("parcel_table_container", "children"),
    [Input("dropdown_category","value"),
     Input("dropdown_segm_methods", "value"),
     Input("parcel_table_dropdown_mode", "value"),
     Input("parcel_table_dropdown_method", "value"),
     Input("parcel_table_dropdown_scalar", "value")])
def change_parcel_table_mode(mode, segmentation_method, table_mode, parcellation_method, scalar):
    
    if table_mode == 'subjects':
        mode = 'subjects'

    return [build_parcel_table(mode = mode, 
                segmentation_method = segmentation_method, 
                parcellation_method = parcellation_method,
                scalar = scalar,
                color = False)]

# Highlight table row borders upon clicking
@app.callback(
    Output("segm_table", "style_data_conditional"),
    [Input("segm_table", "selected_cells")],
    [State("segm_table", "data"),
     State('segm_table', 'style_data_conditional')])
def paint_segm_table(selected_cells, table_data, style_data_conditional):
 
    n_rows = len(table_data)

    for row in range(n_rows):
        rule = {"if": {"row_index": row}, "border": "3px solid blue"}
        if rule in style_data_conditional:
            style_data_conditional.remove(rule)

    if selected_cells is not None:

        for cell in selected_cells:
            row = cell['row'] 
            style_data_conditional.append({"if": {"row_index": row},
                                           "border": "3px solid blue"})
    
    return style_data_conditional

# Highlight table row borders upon clicking
@app.callback(
    Output("parcel_table", "style_data_conditional"),
    [Input("parcel_table", "selected_cells")],
    [State("parcel_table", "data"),
     State('parcel_table', 'style_data_conditional')])
def paint_parcel_table(selected_cells, table_data, style_data_conditional):
    n_rows = len(table_data)

    for row in range(n_rows):
        rule = {"if": {"row_index": row}, "border": "3px solid blue"}
        if rule in style_data_conditional:
            style_data_conditional.remove(rule)

    if selected_cells is not None:

        for cell in selected_cells:
            row = cell['row'] 
            style_data_conditional.append({"if": {"row_index": row},
                                           "border": "3px solid blue"})
    
    return style_data_conditional

@app.callback(
    Output("parcel_download_all", "data"), 
    [Input("parcel_download_all_btn", "n_clicks")],
    [State("dropdown_segm_methods", "value")])
def download_parceldata(n_clicks, segmentation_method): 
    return export_parcel_table(segmentation_method, 'Witelson')
'''

@app.callback(
    [Output('scatter_plot_2', 'figure'),
     Output('scatter_matrix_2', 'figure')],
    [Input('scatter_plot', 'selectedData'),
     Input('scatter_matrix', 'selectedData')],
    [State('scatter_plot', 'figure'),
     State('scatter_matrix', 'figure')])
def get_selected_data(selection1, selection2, fig1, fig2):
    
    if fig1 is None:
        raise PreventUpdate
    else:

        selectedpoints = []
        for selected_data in [selection1, selection2]:
            if selected_data and selected_data['points']:
                selectedpoints = selectedpoints + [p['customdata'] for p in selected_data['points']]
        
        if fig1 is not None and fig2 is not None:
            fig1 = go.Figure(fig1)
            fig1.update_traces(selectedpoints=selectedpoints, customdata=list(path_dict.keys()))
            
            fig2 = go.Figure(fig2)
            fig2.update_traces(selectedpoints=selectedpoints, customdata=list(path_dict.keys()))
        
        return fig1, fig2
'''
'''
@app.callback(
    Output('scatter_matrix', 'figure'),
    [Input('scatter_matrix', 'selectedData'),
     Input('scatter_plot', 'selectedData')],
    [State("dropdown-scalars-right","value"),
     State("dropdown-scalars-left","value")])
def update_selected_points(selection1, selection2, scalar_x, scalar_y):
    selected_points = []
    for selected_data in [selection1, selection2]:
        if selected_data is not None:
            for point_dict in selected_data['points']:
                print(point_dict)
                selected_points.append(point_dict['pointIndex'])
    if selected_points == []:
        selected_points = None
    return build_segm_scattermatrix(selected_points=selected_points)
'''

'''
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
], className = "eight columns"),
'''

# SERVER CONFIG ---------------------------------------------------------------------------------

#port = 5000 + random.randint(0, 999)
port = 5050

def open_browser():
    url = "http://127.0.0.1:{0}".format(port)
    webbrowser.open(url)

if __name__ == "__main__":
    Timer(1.25, open_browser).start()
    app.run_server(debug=False, port=port)
