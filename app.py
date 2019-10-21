import plotly as py
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import base64
import datetime,gzip,pickle
import io,os,pandas as pd
import numpy as np
from scipy import stats


def summary_df(df,type):
    if type == 'cat':
        summary = df.describe(include=[np.object]).T
    else:
        summary = df.describe(exclude=[np.object]).T
        summary.replace([np.nan,np.inf,-np.inf], '', inplace=True)
    summary.reset_index(inplace=True)
    return summary




## Function to reduce the DF size


# os.chdir('C:\\Users\\sudheermoguluri\\CourseEra\\Project\\Fraud_detection')

# def read_initial_data():
#     global train, test
with gzip.open('./train_test_reduced', 'rb') as f:
    train = pickle.load(f)
    test = pickle.load(f)

print('Data loaded successfully.')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# @app.server.before_first_request(read_initial_data())

# @app.server.after_request(summarised_data())

def generate_table(dataframe, max_rows=10):
    return dash_table.DataTable(
            data=dataframe.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in dataframe.columns],
            style_table={'overflowX': 'scroll',
                         'maxHeight': '300',
                         'overflowY': 'scroll'
                         },
            fixed_rows={'headers': True, 'data': 0},
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_cell={
                'height': 'auto',
                # 'minWidth': '0px', 'maxWidth': '180px',
                'width': '90px',
                'whiteSpace': 'normal'
            }
    )

def generate_table_old(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

######## Cat plot #############
uniq_df = pd.DataFrame(columns=['train','test','max_values','equal_values','test_in_train'])
uniq_df = uniq_df.astype({'train':'int','test':'int','max_values':'int','equal_values':'bool','test_in_train':'bool'})
for c in test.columns:
    train_unique = train[c].unique()
    test_unique = test[c].unique()
    max_values = len(pd.Series(list(train_unique) + list(test_unique)).unique())
    if max_values < 30:
        row = pd.Series({
            'train': len(train_unique),
            'test': len(test_unique),
            'max_values': max_values,
            'equal_values': np.array_equal(np.sort(train[c].dropna().unique()), np.sort(test[c].dropna().unique())),
            'test_in_train': all([i in train[c].dropna().unique() for i in test[c].dropna().unique()])
        }, name = c)
        uniq_df = uniq_df.append(row)

cols = uniq_df.query('test_in_train == True').sort_values(by='max_values').index
id_cols = [c for c in cols if c.startswith('id_')]
m_cols = [c for c in cols if c.startswith('M')]
v_cols = [c for c in cols if c.startswith('V')]
o_cols = [c for c in cols if c not in id_cols + m_cols + v_cols]

######## plot for categorical values #########
def cat_plot(columns, plot_name):
    mask = [False] * len(columns)
    mask = mask + mask

    fraud_lbl = {0:'Non Fraud',1:'Fraud'}

    traces = []
    buttons = [{
        "args": ["visible", mask],
        "label": 'Column',
        "method": "restyle"
    }]

    for i, col in enumerate(columns):
        for f in [0,1]:
            query = train.query(f'isFraud == {f}')[col].fillna('NaN').value_counts(dropna=False)
            trace = go.Bar(
                x=query.index.tolist(),
                y=query.tolist(),
                orientation='v',
                name=fraud_lbl[f],
                visible=False
            )
            traces.append(trace)

        mask_temp = mask.copy()
        mask_temp[i*2] = True
        mask_temp[i*2+1] = True
        button = {
            "args": ["visible", mask_temp],
            "label": col,
            "method": "restyle"
        }
        buttons.append(button)


    layout = {
        "title": f"Fraud by Categorical features ({plot_name})",
        'xaxis_type':'category',
        "updatemenus": [{
            "buttons": buttons,
            "yanchor": "top",
            "y": 1.12,
            "x": 0.085
        }]
    }

    return go.Figure(data=traces,layout=layout)

######### historgram plot ###########
def hist_plot(df):
    df_null_hist = (df.isnull().sum() / df.shape[0] * 100).astype(int)
    data = [
        go.Histogram(
            x=df_null_hist,
            nbinsx=25,
            name='df',
            marker_color='#330C73'
        )
    ]
    layout = {
        'title': 'Percentage of Null values in columns',
        'xaxis_title_text': 'Percenteg NaN values',
        'yaxis_title_text': 'Columns count'
    }
    return go.Figure(
        data=data,
        layout=layout
    )

###### number of null columns #########

Train_NaN_cols_count = (train.isna().sum() > 0).sum()
Test_NaN_cols_count = (test.isna().sum() > 0).sum()
Train_cols_count = len(train.columns)
Test_cols_count = len(test.columns)

data_null = [
    go.Bar(
        y=['Train', 'Test'],
        x=[Train_NaN_cols_count, Test_NaN_cols_count],
        type = 'bar',
        name = 'Null',
        orientation='h'
    ),
    go.Bar(
        y=['Train', 'Test'],
        x=[Train_cols_count - Train_NaN_cols_count - 1, Test_cols_count - Test_NaN_cols_count],
        type = 'bar',
        name = 'Not Null',
        orientation='h'
    )
]
layout_null = {
    'barmode': 'relative',
    'title': 'Null columns',
    'xaxis_title_text': 'Count',
    'height': 300
}
fig_null = go.Figure(
    data=data_null,
    layout = layout_null
)


### Reading images
def encoded_image(filename):
    import base64
    dist_plot = base64.b64encode(open(filename, 'rb').read())
    return dist_plot

tab1_layout = [
        html.H4(children='IEEE Fraud Detection - Train data'),
        html.H4(children='Train data - Numeric data'),
        generate_table(summary_df(train,'num')),
        html.H4(children='Train data - Categorical data'),
        generate_table(summary_df(train,'cat')),
        html.H4(children='Fraud Imbalance'),
        dcc.Graph(figure=go.Figure(go.Pie(
            labels=['NoN Fraud', 'Fraud'],
            values=train['isFraud'].value_counts(),
            marker = dict(colors = ['#1499c7',' #f5b041']),
            textinfo='value+percent',
            pull=.03
        ))),
        dcc.Graph(figure=cat_plot(o_cols, 'Other Columns')),
        dcc.Graph(figure=cat_plot(id_cols, 'Id Columns')),
        dcc.Graph(figure=cat_plot(m_cols, 'M Columns')),
        dcc.Graph(figure=cat_plot(v_cols, 'V Columns')),
        dcc.Graph(figure=hist_plot(train)),
        dcc.Graph(figure=fig_null),
        html.H1(children='Log Transformation Amount'),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image(os.getcwd() + '\\plots_images\\dist_plot_log_trx_amt.png').decode())),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image(os.getcwd() + '\\plots_images\\card4_card6.png').decode())),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image(os.getcwd() + '\\plots_images\\dev_type.png').decode())),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image(os.getcwd() + '\\plots_images\\fraud_by_amt_count.png').decode())),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image(os.getcwd() + '\\plots_images\\fraud_by_prod_code.png').decode())),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image(os.getcwd() + '\\plots_images\\prd_cd_train_test.png').decode()))

        ]

tab2_layout = [
        html.H4(children='IEEE Fraud Detection - Test data'),
        html.H4(children='Test data - Numerical data'),
        generate_table(summary_df(test,'num')),
        html.H4(children='Test data - Categorical data'),
        generate_table(summary_df(test, 'cat')),
        dcc.Graph(figure=fig_null),
        dcc.Graph(figure=hist_plot(test)),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image(os.getcwd() + '\\plots_images\\prd_cd_train_test.png').decode())),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image(os.getcwd() + '\\plots_images\\dev_type.png').decode()))
        ]

app.layout = html.Div([
    # html.Div(dash_table.DataTable),
    html.H1('IEEE Fraud Detection'),
    # html.Div(dash_table.DataTable(data=[{}]), style={'display': 'none'}),
    dcc.Tabs(id="tabs", value='tabs', children=[
        dcc.Tab(label='Training-data', value='Training-data',children=tab1_layout),
        dcc.Tab(label='Testing-data', value='Testing-data',children=tab2_layout)
    ])
# html.Div(id='tabs-content')
])

# tab1_layout = html.Div([
#         html.H4(children=['IEEE Fraud Detection - Train data',
#         generate_table(train_sum)]),
# #     dcc.Graph(figure=go.Figure(go.Pie(
# #     labels=['NoN Fraud', 'Fraud'],
# #     values=train['isFraud'].value_counts(),
# #     marker = dict(colors = ['#1499c7',' #f5b041']),
# #     textinfo='value+percent',
# #     pull=.03
# # ))),
# html.Div(id='Training-Data-content')
# ])

# tabs call back
# @app.callback(Output('tabs-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'Training-data':
        return html.Div(children=[
        # html.H4(children='IEEE Fraud Detection - Train data'),
        generate_table(train)
        ])
#     dcc.Graph(figure=go.Figure(go.Pie(
#     labels=['NoN Fraud', 'Fraud'],
#     values=train['isFraud'].value_counts(),
#     marker = dict(colors = ['#1499c7',' #f5b041']),
#     textinfo='value+percent',
#     pull=.03
# ))),
    elif tab == 'Testing-data':
        return html.Div(children=[
        # html.H4(children='IEEE Fraud Detection - Test data'),
        generate_table(test)
])


# tab-1 callback


# def parse_contents(contents, filename, date):
#     content_type, content_string = contents.split(',')
#
#     decoded = base64.b64decode(content_string)
#     try:
#         if 'csv' in filename:
#             # Assume that the user uploaded a CSV file
#             df = pd.read_csv(
#                 io.StringIO(decoded.decode('utf-8')))
#         elif 'xls' in filename:
#             # Assume that the user uploaded an excel file
#             df = pd.read_excel(io.BytesIO(decoded))
#     except Exception as e:
#         print(e)
#         return html.Div([
#             'There was an error processing this file.'
#         ])
#
#     return html.Div([
#         html.H5(filename),
#         html.H6(datetime.datetime.fromtimestamp(date)),
#
#         dash_table.DataTable(
#             data=df.to_dict('records'),
#             columns=[{'name': i, 'id': i} for i in df.columns]
#         ),
#
#         html.Hr(),  # horizontal line
#
#         # For debugging, display the raw contents provided by the web browser
#         html.Div('Raw Content'),
#         html.Pre(contents[0:200] + '...', style={
#             'whiteSpace': 'pre-wrap',
#             'wordBreak': 'break-all'
#         })
#     ])


# @app.callback(Output('output-data-upload', 'children'),
#               [Input('upload-data', 'contents')],
#               [State('upload-data', 'filename'),
#                State('upload-data', 'last_modified')])


# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True, port=8050, threaded=True)