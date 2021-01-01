 #!/usr/bin/env python -W ignore::DeprecationWarning

import os
import ast
import pathlib
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from skimage import measure

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.captureWarnings(True)

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

class Error(Exception):
    pass

theme = 'plotly'

print(' ')

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

if opts.staple is True:
    dict_segmentation_functions['STAPLE'] = segmfuncs.segm_staple
    dict_segmentation_methods['STAPLE'] = 'staple'

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
if opts.folders is not None:
    for directory in opts.folders:
        if directory is not None:
            if inputfuncs.check_directory(directory, opts.basename):
                path_dict[os.path.basename(directory)] = os.path.join(directory, '')

# Import the subjects inside the parents folders
group_dict = {}
if opts.parents is not None:
    for parent in opts.parents:
        if parent is not None:
            
            directory_dict, dict_folders = inputfuncs.import_parent(parent, opts.basename)
            path_dict.update(directory_dict)

            # Create dict with subjects as keys and group (parents names) as values
            group_dict.update(dict_folders)


df_group = pd.DataFrame.from_dict(group_dict, orient='index', columns=["Folder"])
df_categories = pd.concat([df_categories, df_group], axis = 1)

# Warning for no imported subjects
if len(path_dict.values()) == 0:
    print('Error: No subjects were imported')
    print('Terminating program.\n')
    raise SystemExit(0)

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
dict_removed_subjects = {}

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
            dict_removed_subjects[segmentation_method] = []


        # Get data path info
        folderpath = subject_path + 'inCCsight/'
        filename = 'segm_' + dict_segmentation_methods[segmentation_method] + '_data.npy'
        subject_name = os.path.basename(os.path.dirname(subject_path))

        # Process/Load data
        try:
            segmentation_mask, scalar_maps, scalar_statistics, scalar_midlines, error_prob, parcellations_masks = ccprocess.segment(subject_path, 
                                                                                                                                   segmentation_method, 
                                                                                                                                   dict_segmentation_functions, 
                                                                                                                                   dict_parcellation_functions, 
                                                                                                                                   opts.basename)
        except:
            print('Segmentation failed for subject {} with method {}'.format(subject_name, segmentation_method))

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
    
loaded_subjects = list(set(loaded_subjects))
loaded_subjects.sort()

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
app.title = 'inCCsight'


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
                df = df.drop(dict_removed_subjects[segmentation_method])
                
                subplots.add_trace(go.Box(y=df[scalar], name=segmentation_method, legendgroup=segmentation_method, hovertext=df.index, marker=dict(color=std_colors[j])), row=1, col=i+1)

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
        df = df.drop(dict_removed_subjects[segmentation_method])
        df = df.dropna(axis=0)
        
        categories = set(df[mode])
        
        for i, scalar in enumerate(scalar_names):
            
            for j, category in enumerate(categories):
                
                subplots.add_trace(go.Box(y=df[df[mode] == category][scalar], name=category, legendgroup=category, hovertext=df.index, marker=dict(color=std_colors[j])), row=1, col=i+1)

                if i == 0:
                    subplots.data[-1].update(name=category, legendgroup=category)
                else:
                    subplots.data[-1].update(showlegend=False)

    subplots.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h")
    subplots.update_layout(font=dict(family="Open Sans, sans-serif", size=12), margin=dict(r=0, l=0, t=60))
    
    return subplots

def build_parcel_boxplot(scalar='FA', mode='Method', segmentation_method='ROQS', parcellation_method='Witelson'):

    std_colors = pio.templates[theme]['layout']['colorway']
    
    list_regions = ['P1', 'P2', 'P3', 'P4', 'P5']
    subplots = make_subplots(rows=1, cols=5, subplot_titles=list_regions)
    
    if mode == 'Method':
        
        for i, region in enumerate(list_regions):
            
            for j, segmentation_method in enumerate(dict_segmentation_methods.keys()):
                df = dict_parcellations_statistics[segmentation_method][parcellation_method][region][scalar]
                df = df.drop(dict_removed_subjects[segmentation_method])
                df = df.dropna(axis=0)
                subplots.add_trace(go.Box(y=df, name=segmentation_method, legendgroup=segmentation_method, hovertext=df.index, marker=dict(color=std_colors[j])), row=1, col=i+1)

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
        
        df = df.drop(dict_removed_subjects[segmentation_method])
        df = df.dropna(axis=0)

        names = list_regions + [mode]
        df.columns = names

        categories = set(df[mode])
        
        for i, region in enumerate(list_regions):
            
            for j, category in enumerate(categories):
                
                subplots.add_trace(go.Box(y=df[df[mode] == category][region], name=category, legendgroup=category, hovertext=df.index, marker=dict(color=std_colors[j])), row=1, col=i+1)

                if i == 0:
                    subplots.data[-1].update(name=category, legendgroup=category)
                else:
                    subplots.data[-1].update(showlegend=False)

    subplots.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h")
    subplots.update_layout(font=dict(family="Open Sans, sans-serif", size=12), margin=dict(r=0, l=0, t=60))
    
    return subplots

def build_segm_scatterplot(mode='Method', segmentation_method = 'ROQS', scalar_x = 'FA', scalar_y = 'MD', trendline=None):

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
        
    df = df.drop(dict_removed_subjects[segmentation_method])
    df = df.dropna(axis=0)

    fig = px.scatter(df, 
                    x=scalar_x, 
                    y=scalar_y, 
                    color=mode, 
                    marginal_y="violin", 
                    marginal_x="histogram",
                    hover_name=df.index,
                    trendline=trendline)

    fig.update_layout(height=800, paper_bgcolor='rgba(0,0,0,0)')
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

    df = df.drop(dict_removed_subjects[segmentation_method])
    df = df.dropna(axis=0)

    fig = px.scatter_matrix(df, 
                            dimensions=dimensions,
                            color=mode,
                            hover_name=df.index)

    if mode == 'Method':
        n_cats = len(dict_segmentation_methods.keys())
    else:
        n_cats = len(set(df_categories[mode].dropna(axis=0)))

    fig.update_layout(height=250*n_cats, paper_bgcolor='rgba(0,0,0,0)',legend_orientation="h")
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
        df = df.drop(dict_removed_subjects[segmentation_method])
        df = df.dropna(axis=0)
            
        df_grouped = df.groupby('Method').mean().transpose()
        df_melt = pd.melt(df_grouped.reset_index(), id_vars='index', value_vars=set(df[mode]))
    
    else:
        if scalar in ['FA', 'MD', 'AD', 'RD']:
            df_aux = pd.DataFrame.from_dict(dict(dict_scalar_midlines[segmentation_method][scalar]), orient='index', columns=scalar_midline_names)
        elif scalar == 'Thickness':
            df_aux = dict_thickness[segmentation_method]
            
        df = pd.concat([df_categories[mode], df_aux], axis=1)
        df = df.drop(dict_removed_subjects[segmentation_method])
        df = df.dropna(axis=0)

        df_grouped = df.groupby(mode).mean().transpose()
        df_melt = pd.melt(df_grouped.reset_index(), id_vars='index', value_vars=set(df[mode]))
        
    fig = px.line(df_melt, x='index', y='value', color=mode)

    fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h")    
    fig.update_layout(font=dict(family="Open Sans, sans-serif", size=12))
    
    fig.update_layout(xaxis_title='Points along CC body', yaxis_title=scalar, legend_title=mode)
    
    return fig

def build_bubble_grouped(mode='Method', segmentation_method='ROQS', scalar='FA', size=False):
    
    def build_bubble_plot(scalar='FA', segmentation_method='Watershed', size = True, category_index = None):

        df_pts = pd.read_pickle('./assets/bubble_plot_xy.pkl')
        segm_contour = np.load('./assets/bubble_plot_contour.npy')

        if scalar == 'Thickness':
            df_aux = dict_thickness[segmentation_method][list(np.linspace(0,195,40))+[199]]    
        else:
            df_aux = pd.DataFrame.from_dict(dict(dict_scalar_midlines[segmentation_method][scalar]), orient='index', columns=scalar_midline_names)[list(np.linspace(0,195,40))+[199]]
        
        df_aux = df_aux.drop(dict_removed_subjects[segmentation_method])

        if category_index is not None:
            df_aux = df_aux.loc[category_index]

        df_aux = df_aux.dropna(axis=0)

        df_pts[scalar] = df_aux.mean().reset_index()[0]

        if size == True:
            fig = px.scatter(df_pts, x="x", y="y", color = scalar, size = df_pts[scalar])
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
        fig = make_subplots(rows=n_cats, cols=1, vertical_spacing=0.1/n_cats)
        
        for i, segmentation_method in enumerate(dict_segmentation_methods):
            fig.add_trace(build_bubble_plot(scalar=scalar, segmentation_method=segmentation_method, size=size)['data'][0], row=i+1, col=1)
            fig.update_yaxes(title_text="{} for {} method".format(scalar, segmentation_method), row=i+1, col=1)
        
    else:
        
        df = df_categories[mode]
        df = df.drop(dict_removed_subjects[segmentation_method])
        df = df.dropna(axis=0)

        n_cats = len(set(df))
        fig = make_subplots(rows=n_cats, cols=1, vertical_spacing=0.1/n_cats)
        
        for i, category in enumerate(set(df)):
            category_index = df_categories.loc[df_categories[mode] == category].index
            fig.add_trace(build_bubble_plot(scalar=scalar, segmentation_method=segmentation_method, size=size, category_index=category_index)['data'][0], row=i+1, col=1)
            fig.update_yaxes(title_text="{} for {} category".format(scalar, category), row=i+1, col=1)
    
    fig.update_xaxes(title_text="Points along CC body", row=n_cats, col=1)
    fig.update_layout(height=250*n_cats, paper_bgcolor='rgba(0,0,0,0)') 
    fig.update_layout(font=dict(family="Open Sans, sans-serif", size=12), margin=dict(r=0, l=0, t=60))
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
        
        fig = make_subplots(rows=n_cats, cols=1, x_title='Statistic Meaningful Differences (p < 0.05)')
        
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

def build_midlineplot_dropdown():

    options = [{'label': scalar, 'value': scalar} for scalar in scalar_list+['Thickness']]

    layout = html.Div([
                html.Div([
                    html.H6('Scalar:', className='table-options-title', style={'padding':'0px 10px 0px 10px'}),
                    dcc.Dropdown(id='dropdown-midline-scalars',
                                 options=options,
                                 multi=False,
                                 value='FA',
                                 style={'width':'120px'}),
                ], className='row', style=dict(display='flex', justifyContent='left', verticalAlign='center')),
            ]
        )
    return layout

def build_segm_scatterplot_dropdowns():

    options = [{'label': scalar, 'value': scalar} for scalar in scalar_list]
    options_trendlines = [{'label': scalar, 'value': scalar} for scalar in ['None', 'OLS', 'Lowess']]

    layout = html.Div([
                html.Div([
                    html.Div([
                        html.H6('Scalar Y:', className='table-options-title', style={'padding':'0px 10px 0px 0px'}),
                        dcc.Dropdown(id='dropdown-scalars-left',
                                     options=options,
                                     multi=False,
                                     value='FA',
                                     style={'width':'90px'}),
                    ], className='row', style={'margin':'0px 0px 0px 10px'}),
                    html.Div([
                        html.H6('Scalar X:', className='table-options-title', style={'padding':'0px 10px 0px 0px'}),
                        dcc.Dropdown(id='dropdown-scalars-right',
                                     options=options,
                                     multi=False,
                                     value='MD',
                                     style={'width':'90px'}),
                    ], className='row', style={'margin':'0px 0px 0px 30px'}),
                    html.Div([
                        html.H6('Trendline:', className='table-options-title', style={'padding':'0px 10px 0px 0px'}),
                        dcc.Dropdown(id='dropdown-scalars-trendline',
                                     options=options_trendlines,
                                     multi=False,
                                     value='None',
                                     style={'width':'120px'}),

                    ], className='row', style={'margin':'0px 0px 0px 30px'}),
                ], className='row', style=dict(display='flex', justifyContent='left', verticalAlign='center')),
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

def build_bubbleplot_dropdowns():

    options_scalars = [{'label': scalar, 'value': scalar} for scalar in scalar_list+['Thickness']]
    options_size = [{'label': scalar, 'value': scalar} for scalar in ['True', 'False']]

    layout = html.Div([
                html.Div([
                    html.H6('Scalar:', className='table-options-title', style={'padding':'0px 10px 0px 10px'}),
                    dcc.Dropdown(id='dropdown-bubbleplot-left',
                                 options=options_scalars,
                                 multi=False,
                                 value='FA',
                                 style={'width':'150px'}),
                    html.H6('Size:', className='table-options-title', style={'padding':'0px 10px 0px 30px'}),
                    dcc.Dropdown(id='dropdown-bubbleplot-right',
                                 options=options_size,
                                 multi=False,
                                 value='False',
                                 style={'width':'120px'})
                ], className='row', style=dict(display='flex', justifyContent='left', verticalAlign='center')),
            ]
        )
    return layout    

def build_fissure_image_dropdown():

    options = [{'label': segmentation_method, 'value': segmentation_method} for segmentation_method in list(dict_segmentation_methods.keys())+['None']]
    options_scalars = [{'label': scalar, 'value': scalar} for scalar in ['wFA']+scalar_list]

    layout = html.Div([
                html.Div([
                    html.H6('Segm. Method:', className='table-options-title', style={'padding':'0px 10px 0px 10px'}),
                    dcc.Dropdown(id='dropdown-subj-collapse-segm-methods',
                                 options=options,
                                 multi=False,
                                 value=list(dict_segmentation_methods.keys())[0],
                                 style={'width':'120px'}),
                    html.H6('Scalar:', className='table-options-title', style={'padding':'0px 10px 0px 30px'}),
                    dcc.Dropdown(id='dropdown-subj-collapse-scalars',
                                 options=options_scalars,
                                 multi=False,
                                 value='wFA',
                                 style={'width':'150px'})
                ], className='row', style=dict(display='flex', justifyContent='left', verticalAlign='center')),
            ]
        )

    return layout   

def build_individual_subject_segm_table_dropdown():

    options = [{'label': segmentation_method, 'value': segmentation_method} for segmentation_method in dict_segmentation_methods.keys()]
    options_stddev = [{'label': scalar, 'value': scalar} for scalar in ['Show', 'Hide']]

    layout = html.Div([
                html.Div([
                    html.H6('Segm. method:', className='table-options-title', style={'padding':'0px 10px 0px 10px'}),
                    dcc.Dropdown(id='dropdown-subj-collapse-table-segm-methods',
                                 options=options,
                                 multi=False,
                                 value=list(dict_segmentation_methods.keys())[0],
                                 style={'width':'150px'}),
                    html.H6('Show Std.Dev.:', className='table-options-title', style={'padding':'0px 10px 0px 30px'}),
                    dcc.Dropdown(id='dropdown-subj-collapse-table-segm-std-dev',
                                 options=options_stddev,
                                 multi=False,
                                 value='Hide',
                                 style={'width':'120px'})
                ], className='row', style=dict(display='flex', justifyContent='left', verticalAlign='center')),
            ]
        )

    return layout

def build_individual_subject_parcel_table_dropdown():

    options = [{'label': segmentation_method, 'value': segmentation_method} for segmentation_method in dict_segmentation_methods.keys()]
    options_scalars = [{'label': scalar, 'value': scalar} for scalar in scalar_list]
    options_parcel_method = [{'label': parc, 'value': parc} for parc in dict_parcellations_statistics[segmentation_method].keys()]

    layout = html.Div([
                html.Div([
                    html.H6('Segm. method:', className='table-options-title', style={'padding':'0px 10px 0px 10px'}),
                    dcc.Dropdown(id='dropdown-subj-collapse-table-parcel-segm-methods',
                                 options=options,
                                 multi=False,
                                 value=list(dict_segmentation_methods.keys())[0],
                                 style={'width':'150px'}),
                    html.H6('Parcel. method:', className='table-options-title', style={'padding':'0px 10px 0px 30px'}),
                    dcc.Dropdown(id='dropdown-subj-collapse-table-parcel-methods',
                                 options=options_parcel_method,
                                 multi=False,
                                 value=list(dict_parcellation_methods.keys())[0],
                                 style={'width':'150px'}),
                    html.H6('Scalar:', className='table-options-title', style={'padding':'0px 10px 0px 30px'}),
                    dcc.Dropdown(id='dropdown-subj-collapse-table-parcel-scalars',
                                 options=options_scalars,
                                 multi=False,
                                 value='FA',
                                 style={'width':'120px'})
                ], className='row', style=dict(display='flex', justifyContent='left', verticalAlign='center')),
            ]
        )
    return layout 

# -----------------------------------------------------------------------------------------

def build_subjects_list():
    '''
    button_group = dbc.ButtonGroup(
        [dbc.Button(i, color='light', size='lg', style=dict(width='100%')) for i in loaded_subjects],
        vertical=True,
        style=dict(width='100%')
    )
    '''

    button_group = []
    for i, subject_id in enumerate(loaded_subjects):

        if i%2 == 0:
            background = 'rgba(50,50,70,.5)'
        else:
            background = 'rgba(60,60,70,.5)'

        button_group.append(html.Button(subject_id, 
            style=dict(fontSize='1.8rem', width='100%', backgroundColor=background, marginBottom='2px', color='rgb(255,255,255)'), 
            id={'type': 'subject-list-btns', 'index': 'btn-subj-list-{}'.format(subject_id)}))
        

    return html.Div(button_group, style=dict(width='100%'))

def build_quality_collapse():

    layout = html.Div([
                
                html.Div([
                    build_graph_title("Quality evaluation"),
                    html.Button('X', style=dict(fontSize='1.5rem', margin='10px', padding='0 13px', fontFamily= 'Open Sans', borderRadius='20px'),
                                id='btn-exit-quality')
                
                ], className='twelve columns', style=dict(display='flex', justifyContent='space-between')),

                html.Div([
                        html.Div([
                            html.H6("Threshold:"), 
                            dcc.Dropdown(id='dropdown-quality-threshold',
                                         options= [{'label': num/100, 'value': num/100} for num in np.arange(95, 5, -5)],
                                         multi=False,
                                         value='0.7',
                                         style={'width':'100px', 'marginLeft':'5px'}),
                            ], className='row', style=dict(marginLeft='2rem')),
                ], className='twelve columns', style=dict(display="flex", justifyContent="space-between")),
                
                html.Div([
                    dbc.Button("Restore Removed", color='info', outline=True, size='lg', id='restore_btn', style=dict(marginRight='1rem')),
                    dbc.Button("Remove Selected", color='danger', outline=True, size='lg', id='remove_btn', style=dict(marginRight='2rem')),
                ], className='twelve columns', style=dict(display='flex', justifyContent='flex-end')),

                html.Div([
                    dbc.Button("Unselect all", color='primary', outline=True, size='lg', id='unselect_all_btn', style=dict(marginRight='1rem')),
                    dbc.Button("Select all", color='primary', outline=True, size='lg', id='select_all_btn', style=dict(marginRight='2rem')),
                ], className='twelve columns', style=dict(display='flex', verticalAlign='center', justifyContent='flex-end', padding='10px 0px 10px 0px', marginBottom='-1rem')),

                html.Div(children=build_quality_images(), 
                    className="twelve columns", 
                    id='photo-container',
                    style=dict(margin="0rem 0rem 2rem 0rem")),
            
            ], style={'margin':'20px', 'height':"100vh", 'backgroundColor':'#FAFAFA', 'border-radius':'20px', 'border':'1px solid rgba(0,0,0,.125)'})

    return layout

def build_quality_images(threshold=0.7):
    
    def get_quality_tab_children(segmentation_method):
        
        children = []

        # Get error probs
        df = dict_error_prob[segmentation_method]
        df = df.drop(dict_removed_subjects[segmentation_method])
        df = df.sort_values(by=['error_prob'])
        
        # Order by error probs
        index_label = df.query('error_prob > '+str(threshold)).index.tolist()

        # Retrieve images and segmentation
        for subject_id in index_label:
            children.append(dcc.Loading(
                            html.Div([
                                html.Div([
                                    html.H6("Subject: {}".format(subject_id), 
                                        style=dict(fontSize='2rem')),
                                    dcc.Checklist(
                                        id={'type': 'remove-cbx', 'index': 'cbx-{}-{}'.format(segmentation_method, subject_id)},
                                        options=[{'label': 'Remove', 'value': 'Remove'}], 
                                        value=[], 
                                        style=dict(fontSize='1.8rem'), inputStyle=dict(marginRight="10px")),  

                                    ], className='twelve columns', style=dict(width='100%', display='flex', verticalAlign='center', justifyContent='space-between')),
                                
                                html.Div([
                                    dcc.Graph(figure=build_fissure_image(subject_id, segmentation_method))
                                    ], className='twlve columns'),
                            
                            ], className = 'twelve columns')))
        return children


    def get_quality_tab(segmentation_method):
        tab = dbc.Tab(label=segmentation_method, children=html.Div(get_quality_tab_children(segmentation_method), style=dict(height='80vh', overflowY="auto", padding='20px 20px 20px 20px')))
        return tab

    tabs = []

    for segmentation_method in dict_segmentation_functions.keys():
        tabs.append(get_quality_tab(segmentation_method))

    return dbc.Tabs(tabs, style=dict(height='40px', verticalAlign='center', padding='0px 10px 0px 10px'))

def build_fissure_image(subject_id, segmentation_method, scalar = 'FA'):

    scalar_maps = dict_scalar_maps['ROQS'][subject_id]
    scalar_maps_list = ['wFA','FA','MD','RD','AD']
    scalar_map = scalar_maps[scalar_maps_list.index(scalar)]

    fig = px.imshow(scalar_map, color_continuous_scale='gray', aspect='auto')

    if segmentation_method != 'None':

        segmentation = dict_segmentation_masks[segmentation_method][subject_id]

        contours = measure.find_contours(segmentation, 0.1)
        contour = sorted(contours, key=lambda x: len(x))[-1]

        fig.add_trace(go.Scatter(x=contour[:, 1], y=contour[:, 0]))
    
    fig.update_layout(height=250, width=450, paper_bgcolor='rgba(0,0,0,0)', legend_orientation="h", coloraxis_showscale=True)
    fig.update_layout(margin = dict(l=0, r=0,t=0,b=30))

    return fig 

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

    layout = dbc.Card([

                html.Div([
                    html.Div([
                        build_graph_title("Subject " + subject_id),
                        dbc.Button(
                                'Remove subject', 
                                outline=True,
                                color='danger',
                                id=dict(type='btn-remove-subject', index=subject_id),
                                style=dict(padding='0 15px', margin='25px 0px 0px 20px', fontSize='1.2rem')
                            ),
                        ], className='row'),
                    html.Button('X', style=dict(fontSize='1.5rem', margin='10px', padding='0 13px', fontFamily= 'Open Sans', borderRadius='20px'),
                                id=dict(type='btn-exit-subject', index=subject_id))
                
                ], className='twelve columns', style=dict(display='flex', justifyContent='space-between')),

                html.Div([                
    
                    # Segmentation image
                    html.Div([
                        build_graph_title("Scalar maps"),
                        #dcc.Graph(figure=build_3d_visualization(subject_id))
                        dcc.Graph(figure=build_fissure_image(subject_id, segmentation_method, scalar = scalar_map), id='subject_collapse_fissure_img'),
                        build_fissure_image_dropdown(),

                    ], className = 'three columns'),

                    html.Div(className='one column'),

                    html.Div([
                        build_graph_title("Segmentation data"),
                        build_individual_subject_segm_table(subject_id, segmentation_method),
                        build_individual_subject_segm_table_dropdown(),

                        build_graph_title("Parcellation data"),
                        build_individual_subject_parcel_table(subject_id, segmentation_method, parcellation_method='Witelson', scalar='FA'),
                        build_individual_subject_parcel_table_dropdown(),

                    ], className='six columns'),

                ], className='row', style={'justifyContent':'center'}),

            ], style={'margin':'20px', 'backgroundColor':'#FAFAFA', 'border-radius':'20px', 'padding':'0px 0px 50px 0px'})
    return layout

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
        df = dict_scalar_statistics[segmentation_method]
        df = df.drop(dict_removed_subjects[segmentation_method], errors='ignore')
        df = df.reset_index().rename(columns={"index": "Subject"})
        names = ['Subject'] + list_scalars
    
    elif mode == 'Method':
        df = pd.DataFrame()
        for segmentation_method in dict_segmentation_functions.keys():
            df_aux = dict_scalar_statistics[segmentation_method]
            df_aux = df_aux.drop(dict_removed_subjects[segmentation_method], errors='ignore')
            df_aux = df_aux.reset_index()
            df_aux['Method'] = segmentation_method
            df = pd.concat([df, df_aux], axis=0)
        df = df.groupby('Method').mean().reset_index()
        names = ['Method'] + list_scalars

    else:
        df = dict_scalar_statistics[segmentation_method]
        df = df.drop(dict_removed_subjects[segmentation_method], errors='ignore')
        df = pd.concat([df_categories[mode], df], axis=1)
        df = df.groupby(mode).mean().reset_index()
        names = [mode] + list_scalars
    
    df = df.round(6).sort_index()

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
        df = df.drop(dict_removed_subjects[segmentation_method], errors='ignore')
        df = df.reset_index()
        df = df.rename(columns={"index": "Subject"})
        names = ['Subject'] + list_regions
    
    elif mode == 'Method':
        
        df = pd.DataFrame()
        for segmentation_method in dict_segmentation_methods.keys():
            df_aux = pd.DataFrame()
            for region in list_regions:
                df_aux = pd.concat([df_aux, dict_parcellations_statistics[segmentation_method][parcellation_method][region][scalar]], axis=1)
            df_aux = df_aux.drop(dict_removed_subjects[segmentation_method], errors='ignore')
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
        df = df.drop(dict_removed_subjects[segmentation_method], errors='ignore')
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

def export_parcel_table(segmentation_method = 'ROQS', parcellation_method = 'Witelson', include_segm=True, include_parc=True, include_cat=True, groupby_cat=None, include_stdev=True):

    # Get data, organize with multiindex columns
    if include_parc is True:
        df = pd.concat(dict_parcellations_statistics[segmentation_method][parcellation_method].values(), 
               axis = 1, 
               keys = dict_parcellations_statistics[segmentation_method][parcellation_method].keys())
    
    # Include info for whole CC
    if include_segm is True:
        df_segm = pd.concat(dict(CC = dict_scalar_statistics[segmentation_method].drop(columns=['Method'])), axis=1)
        
        if include_parc is True:
            df = pd.concat([df_segm, df], axis=1)
        else:
            df = df_segm

    # Remove info from removed subjects
    if groupby_cat is None:
        df.at[dict_removed_subjects[segmentation_method], df.columns] = ''
    else:
        df = df.drop(dict_removed_subjects[segmentation_method])

                
    # Add categories info
    if include_cat is True or groupby_cat is not None:
        df_aux = pd.concat(dict(Categories=df_categories), axis=1)
        df = pd.concat([df_aux, df], axis=1)

    # Group by category
    if groupby_cat is not None:
        df = df.groupby(('Categories', groupby_cat)).mean()    

    # Drop or not Std Dev columns
    if include_stdev is False:
        for col_tuple in df.columns:
            if 'StdDev' in col_tuple[-1]:
                df = df.drop(columns=col_tuple)

    # Define filename
    if include_segm is True and include_parc is False:
        filename = "inCCsight_data_{}.xlsx".format(segmentation_method)
    elif include_segm is False and include_parc is True:
        filename = "inCCsight_data_{}.xlsx".format(parcellation_method)
    elif include_segm is True and include_parc is True:
        filename = "inCCsight_data_{}_{}.xlsx".format(segmentation_method, parcellation_method)
    else:
        filename = "inCCsight_data.xlsx"

    return send_data_frame(df.to_excel, filename)

def build_export_parcel_table_modal():

    options_segm_config = [{'label': item, 'value': item} for item in dict_segmentation_methods.keys()]
    options_parc_config = [{'label': item, 'value': value} for item, value in zip(list(dict_parcellation_methods.keys()), ['Witelson', 'Hofer', 'Chao', 'Cover', 'Freesurfer'])]
    options_data_config = [{'label': item, 'value': item} for item in ['Categories', 'Segmentation', 'Parcellation']]
    options_stdv_config = [{'label': item, 'value': item} for item in ['Show Std. Dev.']]
    
    options_groupby_check = [{'label': item, 'value': item} for item in ['Group by:']]
    options_groupby_config = [{'label': item, 'value': item} for item in df_categories.columns]

    segm_config = dbc.FormGroup(
        row=True,
        children = [
            dbc.Label("Segmentation Method:", width=4, style=dict(justifyContent='flex-end')),
            dbc.Col(
                dcc.Dropdown(
                    id="export_segm_method",
                    options=options_segm_config,
                    value='ROQS'),
                
                width=8)], 
            )

    parc_config = dbc.FormGroup(
        row=True,
        children = [
            dbc.Label("Parcellation Method:", width=4, style=dict(justifyContent='flex-end')),
            dbc.Col(
                dcc.Dropdown(
                    id="export_parc_method",
                    options=options_parc_config,
                    value='Witelson'),
                
                width=8)], 
            )

    data_config = dbc.FormGroup(
        row=True,
        children = [
            dbc.Label("Include data from:", width=4, style=dict(justifyContent='flex-end')),
            dbc.Col(
                dcc.Checklist(
                    id="export_data_config",
                    options=options_data_config,
                    value=['Categories', 'Segmentation', 'Parcellation'],
                    inputStyle=dict(marginRight='5px')),
                
                width=8)], 
            )

    extra_config = dbc.FormGroup(
        row=True,
        children = [
            dbc.Col(
                width=4,
                children = [
                    dcc.Checklist(
                        id="export_show_stdev",
                        options=options_stdv_config,
                        value=['Show Std. Dev.'],
                        inputStyle=dict(marginRight='5px'))
                    ],
                ),

            dbc.Col(
                width=8,
                children = [
                    dcc.Checklist(
                        id="export_groupby_check",
                        options=options_groupby_check,
                        inputStyle=dict(marginRight='5px')),
                    dcc.Dropdown(
                        id="export_groupby_config",
                        options=options_groupby_config,
                        value='Folder'),
                    ],
                )
            ])


    form = dbc.Form([segm_config, parc_config, data_config, extra_config])

    modal = dbc.Modal([
                
                dbc.ModalHeader("Download table data"),
                dbc.ModalBody(form, style=dict(fontSize='14pt')),
                dbc.ModalFooter(
                        children=[
                            dbc.Label(id='disable_download_message', style=dict(marginRight='10px', opacity='0.5')),
                            dbc.Button("Download", id="download_data_btn", size='lg', style=dict(marginRight='10px')),
                            Download(id="parcel_download_all"),
                            dbc.Button("Close", id="close_modal_btn", size='lg')
                            ], 
                        style=dict(justifyContent='flex-end'))
                ], 

                size='lg',
                centered=True,
                id="export_data_modal"),

    return modal

def build_individual_subject_segm_table(subject_id, segmentation_method = 'ROQS', show_stdev=False):

    df = dict_scalar_statistics[segmentation_method].reset_index() 
    df = df[df['index'] == subject_id].rename(columns={'index':'Subject'})
    
    if show_stdev is True:
        names = ['Subject'] + scalar_statistics_names
    else:
        names = ['Subject', 'FA', 'RD', 'AD', 'MD']

    layout = dash_table.DataTable(
        id = 'individual_subject_segm_table',
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

    return html.Div(layout, style=dict(margin='-30px 0px 2px 0px'), id='individual_subject_segm_table_container')

def build_individual_subject_parcel_table(subject_id, segmentation_method = 'ROQS', parcellation_method='Witelson', scalar='FA'):

    list_regions = ['P1', 'P2', 'P3', 'P4', 'P5']

    df = pd.DataFrame()
    for region in list_regions:
        df = pd.concat([df, dict_parcellations_statistics[segmentation_method][parcellation_method][region][scalar]], axis=1)
    df.columns = list_regions
    
    df = df.reset_index()
    df = df[df['index'] == subject_id].rename(columns={'index':'Subject'})
    

    names = ['Subject'] + list_regions

    layout = dash_table.DataTable(
        id = 'individual_subject_parcel_table',
        columns = [{"name": name, "id": name} for name in names],
        data = df.round(6).reset_index().sort_index().to_dict('records'),

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

    return html.Div(layout, style=dict(margin='-30px 0px 2px 0px'), id='individual_subject_parcel_table_container')

# Extra functions

def get_number_of_folders():
    if len(list(group_dict.values())) > 0:
        return len(set(np.hstack(list(group_dict.values()))))
    else:
        return 1

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
                                            "Check quality",
                                            size='lg',
                                            color="light",
                                            id='quality-button',
                                            style=dict(marginRight='20px')
                                        ),
                                        
                                        dbc.Button(
                                            'Download tabled data', 
                                            size='lg',
                                            color="light",
                                            id='parcel_download_all_btn',
                                        ),

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
                                build_graph_title("Subjects"),
                                html.Div(
                                    build_subjects_list(), 
                                    style=dict(height='500px', overflowY='auto'),
                                    id="subject-table-container")], 
                                className = "four columns", 
                                id="subject-list-container",
                            ),

                            html.Div(className="four columns"),

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
                                        html.H1(str(get_number_of_folders()), className='numcard', style={'margin-top': '10px'}),
                                        html.H5("Folders", className='titlecard'),
                                    ],
                                ),
                            ]),

                        ], className="one-half column", id="right-panel"),
                    ],
                ),
            ],
        ),
        
        # Dashboard row
        html.Div(
            className='row',
            children=[    
        
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            children=build_quality_collapse(),
                            style=dict(padding='0')
                            ),
                        style=dict(border='1px solid rgba(0,0,0,0)'),
                        ),
                    id="quality-collapse",
                    className='quality columns',
                ),
            
                html.Div(
                    id='dashboard',
                    style=dict(height='100vh', overflowY='auto', overflowX='hidden'),
                    className='twelve columns',
                    children=[

                        dbc.Collapse(
                            dbc.Card(dbc.CardBody()), 
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

                                            ],
                                        ),
                                    ],
                                )
                                

                            ]
                        ),

                        html.Div(
                            className="row",
                            id="bottom-row",
                            style=dict(marginRight='20px'),
                            children=[

                                html.Div(
                                    id="bottom-row-left-column",
                                    className="eight columns",
                                    children=[
                                    
                                        html.Div(
                                            id="boxplot-container1",
                                            style=dict(marginBottom='-30px'),
                                            children=[
                                                build_graph_title("Segmentation BoxPlots"),
                                                dcc.Graph(id="segm_boxplots", figure=build_group_segm_boxplot()),
                                            ],
                                        ),

                                        html.Div(
                                            id="boxplot-container2",
                                            #style=dict(maginTop='0px'),
                                            children=[
                                                build_graph_title("Parcellation BoxPlots"),
                                                dcc.Graph(id="parc_boxplots", figure=build_parcel_boxplot()),
                                                build_parcel_boxplot_dropdowns(),
                                            ],
                                        ),
                                    ],
                                ),
                                
                                # Scatterplot
                                html.Div(
                                    id="scatterplot-container",
                                    className="four columns",
                                    children=[
                                        build_graph_title("Scatter Plot"),
                                        dcc.Graph(id="scatter_plot", figure=build_segm_scatterplot()),
                                        build_segm_scatterplot_dropdowns(),
                                    ],
                                ),
                            ],
                        ),

                        html.Div(
                            className="row",
                            id="second-bottom-row",
                            style=dict(marginRight='20px'),
                            children=[
                            
                                # Scattermatrix
                                html.Div(
                                    id="scattermatrix-container",
                                    className="eight columns",
                                    children=[
                                        build_graph_title("Scatter Matrix"),
                                        dcc.Graph(id="scatter_matrix", figure=build_segm_scattermatrix()),
                                    ],
                                ),

                                # Bubble plot
                                html.Div(
                                    id="bubbleplot-container",
                                    className="four columns",
                                    children=[
                                        build_graph_title("Bubble Plots"),
                                        dcc.Graph(id="bubble_plots", figure=build_bubble_grouped()),
                                        build_bubbleplot_dropdowns(),
                                    ],
                                ),

                            ],
                        ),

                        html.Div(
                            className="row",
                            id="third-bottom-row",
                            style=dict(marginRight='20px'),
                            children=[



                                # Midline plots
                                html.Div(
                                    id="midline-container",
                                    className="four columns",
                                    children=[
                                        build_graph_title("Midline Plots"),
                                        dcc.Graph(id="midline_graph", figure=build_midline_plot()),
                                        build_midlineplot_dropdown(),
                                    ],
                                ),

                                html.Div(build_export_parcel_table_modal(), id='example_div'),
                            ],
                        ),        
                    ]
                )
            ]
        )
    ],
)

# --------------------------------- CALLBACKS ---------------------------------------------

# Banner --------------------------------------------------------------

# Enable/Disable segm method dropdown
@app.callback(
    Output("dropdown_segm_methods","disabled"),
    [Input("dropdown_category","value")])
def update_dropdown_disabled(mode):
    if mode == 'Method':
        return True
    else:
        return False

# Graph updates -------------------------------------------------------

# Update segm box-plots
@app.callback(
    Output("segm_boxplots", "figure"),
    [Input("dropdown_segm_methods","value"),
     Input("dropdown_category","value"),
     Input('photo-container', 'children')])
def update_segm_boxplots(segm_method, mode, removed):
    return build_group_segm_boxplot(mode=mode, segmentation_method=segm_method)

# Update segm box-plots
@app.callback(
    Output("parc_boxplots", "figure"),
    [Input("dropdown_segm_methods","value"),
     Input("dropdown_category","value"),
     Input("dropdown-parcel-boxplot-left","value"),
     Input("dropdown-parcel-scalars-right","value"),
     Input('photo-container', 'children')])
def update_segm_boxplots(segm_method, mode, parc_method, scalar, removed):
    return build_parcel_boxplot(scalar=scalar, mode=mode, segmentation_method=segm_method, parcellation_method=parc_method)

# Update midline plot
@app.callback(
    Output("midline_graph", "figure"),
    [Input("dropdown_segm_methods","value"),
     Input("dropdown_category","value"),
     Input("dropdown-midline-scalars","value"),
     Input('photo-container', 'children')])
def update_midlineplot(segm_method, mode, scalar, removed):
    return build_midline_plot(mode=mode, segmentation_method=segm_method, scalar=scalar)

# Update scatter matrix
@app.callback(
    Output("scatter_matrix", "figure"),
    [Input("dropdown_segm_methods", "value"),
     Input("dropdown_category","value"),
     Input('photo-container', 'children')])
def update_scattermatrix(segm_method, mode, removed):
    return build_segm_scattermatrix(mode=mode, segmentation_method=segm_method)

# Update scatter plot
@app.callback(
    Output("scatter_plot","figure"),
    [Input("dropdown_segm_methods","value"),
     Input("dropdown_category","value"),
     Input("dropdown-scalars-right","value"),
     Input("dropdown-scalars-left","value"),
     Input("dropdown-scalars-trendline","value"),
     Input('photo-container', 'children')])
def update_scatterplot(segm_method, mode, scalar_x, scalar_y, trendline, removed):
    if trendline == 'None':
        trendline = None
    else:
        trendline = trendline.lower()
    return build_segm_scatterplot(mode=mode, segmentation_method=segm_method, scalar_x=scalar_x, scalar_y=scalar_y, trendline=trendline)

# Update bubble plot
@app.callback(
    Output("bubble_plots","figure"),
    [Input("dropdown_segm_methods","value"),
     Input("dropdown_category","value"),
     Input("dropdown-bubbleplot-left", "value"),
     Input("dropdown-bubbleplot-right", "value"),
     Input('photo-container', 'children')])
def update_bubbleplot(segm_method, mode, scalar, size, removed):
    if size == 'True':
        size = True
    elif size == 'False':
        size = False
    return build_bubble_grouped(mode=mode, segmentation_method=segm_method, scalar=scalar, size=size)

# Subject collapse ----------------------------------------------------

# Open subject collapse
@app.callback(
    [Output("subject-collapse", "is_open"),
     Output("subject-collapse", "children")],
    [Input({'type': 'subject-list-btns', 'index': ALL}, 'n_clicks'),
     Input({'type': 'btn-exit-subject', 'index': ALL}, 'n_clicks')],
    [State({'type': 'subject-list-btns', 'index': ALL}, 'id')])
def open_subject_collapse(n_clicks, exit_clicks, ids):
    if n_clicks is not None:

        trigger = dash.callback_context.triggered[0] 
        
        if (trigger['value'] is None): 
            return False, []

        btn_id = ast.literal_eval(trigger["prop_id"][:-9])

        if btn_id['type'] == 'btn-exit-subject':
            return False, []        

        subject_id = btn_id['index'].rsplit('-')[-1]
        
        global selected_subject_id
        selected_subject_id = subject_id
        
        return True, [build_subject_collapse(subject_id = subject_id)]
    else:
        return False, []

@app.callback(
    Output('individual_subject_segm_table_container', 'children'),
    [Input('dropdown-subj-collapse-table-segm-methods', 'value'),
     Input('dropdown-subj-collapse-table-segm-std-dev', 'value')])
def update_segm_table_subj_collapse(segmentation_method, show_stdev):
    if show_stdev == 'Show':
        show_stdev = True
    else:
        show_stdev = False

    return build_individual_subject_segm_table(selected_subject_id, segmentation_method, show_stdev)

@app.callback(
    Output('individual_subject_parcel_table_container', 'children'),
    [Input('dropdown-subj-collapse-table-parcel-scalars', 'value'),
     Input('dropdown-subj-collapse-table-parcel-segm-methods', 'value'),
     Input('dropdown-subj-collapse-table-parcel-methods', 'value')])
def update_parc_table_subj_collapse(scalar, segmentation_method, parcellation_method):
    return build_individual_subject_parcel_table(selected_subject_id, segmentation_method, parcellation_method, scalar)

@app.callback(
    Output('subject_collapse_fissure_img', 'figure'),
    [Input('dropdown-subj-collapse-segm-methods', 'value'),
     Input('dropdown-subj-collapse-scalars', 'value')])
def update_fissure_image(segmentation_method, scalar):
    return build_fissure_image(subject_id=selected_subject_id, segmentation_method=segmentation_method, scalar=scalar)

# Quality collapse ----------------------------------------------------

# Remove selected subjects
@app.callback(
    Output('photo-container', 'children'),
    [Input('remove_btn', 'n_clicks'),
     Input('restore_btn', 'n_clicks'),
     Input('dropdown-quality-threshold', 'value')],
    [State({'type': 'remove-cbx', 'index': ALL}, 'id'),
     State({'type': 'remove-cbx', 'index': ALL}, 'value'),
     State('photo-container', 'children')])
def remove_quality_images(n_clicks, restore_clicks, threshold, ids, values, children):

    global dict_removed_subjects

    trigger = dash.callback_context.triggered[0]
    removed_counter = 0

    if restore_clicks is not None and trigger['prop_id'] == 'restore_btn.n_clicks':

        for segmentation_method in dict_segmentation_methods.keys():
            removed_counter += len(dict_removed_subjects[segmentation_method])
            dict_removed_subjects[segmentation_method] = []

    elif n_clicks is not None and trigger['prop_id'] == 'remove_btn.n_clicks':

        for dict_id, value in zip(ids,values):
            if value == ['Remove']:
                segmentation_method, subject_key = dict_id['index'].rsplit('-')[-2:]
                dict_removed_subjects[segmentation_method].append(subject_key)
                dict_removed_subjects[segmentation_method] = list(set(dict_removed_subjects[segmentation_method]))
                removed_counter += 1

    if removed_counter > 0:
        return build_quality_images(threshold)
    else:
        return dash.no_update

# (Un)Select all subjects (check all remove checkboxes)
@app.callback(
    Output({'type': 'remove-cbx', 'index': ALL}, 'value'),
    [Input('select_all_btn', 'n_clicks'),
     Input('unselect_all_btn', 'n_clicks')],
    [State({'type': 'remove-cbx', 'index': ALL}, 'value')])
def select_all_subjects(select_clicks, unselect_clicks, list_bts):
    
    trigger = dash.callback_context.triggered[0]

    if select_clicks is not None and trigger['prop_id'] == 'select_all_btn.n_clicks':
        return [['Remove']] * len(list_bts)

    elif unselect_clicks is not None and trigger['prop_id'] == 'unselect_all_btn.n_clicks':
        return [[]] * len(list_bts)

    else:
        return dash.no_update

# Open quality collapse
@app.callback(
    [Output("dashboard", "className"),
     Output("quality-collapse","is_open")],
    [Input("quality-button","n_clicks"),
     Input("btn-exit-quality","n_clicks")],
    [State("dashboard", "className")])
def open_quality_collapse(n_clicks, exit_clicks, className):
    if n_clicks is not None:
        trigger = dash.callback_context.triggered[0]
        if trigger["prop_id"].split(".")[0][-18:-2] == 'btn-exit-quality':
            return ["twelve columns", False]
        else:
            if className == 'notquality columns':
                return ["twelve columns", False]
            elif className == 'twelve columns':
                return ["notquality columns", True]
    else:
        return ["twelve columns", False]

# Tables --------------------------------------------------------------

# Change segm table mode
@app.callback(
    Output("segm_table_container", "children"),
    [Input("segm_table_dropdown_mode", "value"),
     Input("dropdown_segm_methods", "value"),
     Input("dropdown_category","value"),
     Input("segm_table_dropdown_stdev", "value"),
     Input('photo-container', 'children')])
def change_segm_table_mode(table_mode, segmentation_method, mode, show_stdev, removed):
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
     Input("parcel_table_dropdown_scalar", "value"),
     Input('photo-container', 'children')])
def change_parcel_table_mode(mode, segmentation_method, table_mode, parcellation_method, scalar, removed):
    
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

# Open export data modal
@app.callback(
    Output("export_data_modal", "is_open"),
    [Input("parcel_download_all_btn", "n_clicks"),
     Input("close_modal_btn", "n_clicks")])
def toggle_export_modal(open_clicks, close_clicks):
    if open_clicks is not None:
        
        trigger = dash.callback_context.triggered[0]
        if trigger['prop_id'] == 'parcel_download_all_btn.n_clicks':
            return True
        else:
            return False

# Disable download button
@app.callback(
    [Output("download_data_btn", "disabled"),
     Output("disable_download_message", "children")],
    [Input("export_data_config", "value")])
def toggle_download_btn(value):
    if value == []:
        return True, "You must include at least one type of data."
    else:
        return False, ""

@app.callback(
    Output("parcel_download_all", "data"), 
    [Input("download_data_btn", "n_clicks")],
    [State("export_segm_method", "value"),
     State("export_parc_method", "value"),
     State("export_data_config", "value"),
     State("export_groupby_check", "value"),
     State("export_groupby_config", "value"),
     State("export_show_stdev", "value"),
     ])
def download_parceldata(n_clicks, segmentation_method, parcellation_method, data_config, groupby_check, groupby_config, stdev_config):
    if n_clicks is not None:

        include_cat = False
        include_segm = False
        include_parc = False
        include_stdev = False
        groupby_cat = None

        if data_config is not None:
            if 'Categories' in data_config:
                include_cat = True
            if 'Segmentation' in data_config:
                include_segm = True
            if 'Parcellation' in data_config:
                include_parc = True

        if groupby_check == ['Group by:']:
            groupby_cat = groupby_config

        if stdev_config == ['Show Std. Dev.']:
            include_stdev = True

        return export_parcel_table(segmentation_method, parcellation_method, include_segm, include_parc, include_cat, groupby_cat, include_stdev)

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


# SERVER CONFIG ---------------------------------------------------------------------------------

port = opts.port

def open_browser():
    url = "http://localhost:{0}".format(port)
    webbrowser.open(url)

if __name__ == "__main__":
    Timer(1.25, open_browser).start()
    app.run_server(debug=False, port=port, host='0.0.0.0')
