import dash
from dash import dcc, html, Input, Output, dash_table, State
import json
import requests
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import time
import os
import copy
import datetime

def get_barges_from_json(contents):
    df = pd.json_normalize(contents)
    barges = []
    fleet = df['vessels'].values[0]
    for barge in fleet:
        barges.append(barge['id'])
    return barges

def process_input_for_planning_board(input):
    planning = copy.deepcopy(input['routes'])
    appended_data = []
    for v in range(len(planning)):
        stops = planning[v]['stops']
        vessel = planning[v]['vessel']
        df_stops = pd.DataFrame.from_dict(stops)
        df_stops = df_stops.drop(columns=['loadOrders', 'dischargeOrders'])
        df_stops['vessel'] = vessel
        df_stops = df_stops[['vessel'] + [c for c in df_stops if c not in ['vessel']]]
        appended_data.append(df_stops)

    df = pd.concat(appended_data)

    df = df.drop(columns=['reefersOnBoardAfterStop', 'dangerousGoodsOnBoardAfterStop', 'fixedStop', 'fixedAppointment'])
    vessels = df.vessel.unique().tolist()

    for vessel in vessels:
        v = df[df['vessel']==vessel]
        if (len(v) > 1 and sum(v.iloc[0, -6:]) == 0 and v['teuOnBoardAfterStop'].iloc[0] == 0 and v['weightOnBoardAfterStop'].iloc[0] == 0):
            df[df['vessel']==vessel] = df[df['vessel']==vessel].iloc[1:, :]
            df = df.dropna()
        rowsToRemove = []
        scope = list(v.index.values)
        scope.pop(0)
        for i in scope:
            if df.iloc[i-1,1] == df.iloc[i,1] and df.iloc[i-1,3] == df.iloc[i,2]:
                rowsToRemove.append(i)
                df.iloc[i-1,3] = df.iloc[i,3]
                df.iloc[i-1,4] = df.iloc[i,4]
                df.iloc[i-1,5] = df.iloc[i,5]
                df.iloc[i-1,6] = df.iloc[i-1,6] + df.iloc[i,6]
                df.iloc[i-1,7] = df.iloc[i-1,7] + df.iloc[i,7]
                df.iloc[i-1,8] = df.iloc[i-1,8] + df.iloc[i,8]
                df.iloc[i-1,9] = df.iloc[i-1,9] + df.iloc[i,9]
                df.iloc[i-1,10] = df.iloc[i-1,10] + df.iloc[i,10]
                df.iloc[i-1,11] = df.iloc[i-1,11] + df.iloc[i,11]
                if (i+1 <= scope[-1]):
                    ind = scope.index(i)+1
                    scope2 = scope[ind:]
                    for j in scope2:
                        if (df.iloc[i-1,1] == df.iloc[j,1] and df.iloc[i-1,3] == df.iloc[j,2]):
                            df.iloc[i-1,3] = df.iloc[j,3]
                            df.iloc[i-1,4] = df.iloc[j,4]
                            df.iloc[i-1,5] = df.iloc[j,5]
                            df.iloc[i-1,6] = df.iloc[i-1,6] + df.iloc[j,6]
                            df.iloc[i-1,7] = df.iloc[i-1,7] + df.iloc[j,7]
                            df.iloc[i-1,8] = df.iloc[i-1,8] + df.iloc[j,8]
                            df.iloc[i-1,9] = df.iloc[i-1,9] + df.iloc[j,9]
                            df.iloc[i-1,10] = df.iloc[i-1,10] + df.iloc[j,10]
                            df.iloc[i-1,11] = df.iloc[i-1,11] + df.iloc[j,11]              
    df[df['vessel']==vessel] = df[df['vessel']==vessel].drop(index=rowsToRemove)

    date_format = '%Y-%m-%dT%H:%M:%S'
    df['startTime'] = pd.to_datetime(df['startTime'], format=date_format)
    df['departureTime'] = pd.to_datetime(df['departureTime'], format=date_format)
    df['startTime'] = df['startTime'].astype(str)
    df['departureTime'] = df['departureTime'].astype(str)
    return df

def load_and_process_kpis(input):
    planning = copy.deepcopy(input['routes'])
    appended_data = []
    for v in range(len(planning)):
        stops = planning[v]['stops']
        vessel = planning[v]['vessel']
        df_stops = pd.DataFrame.from_dict(stops)
        df_stops = df_stops.drop(columns=['loadOrders', 'dischargeOrders'])
        df_stops['vessel'] = vessel
        df_stops = df_stops[['vessel'] + [c for c in df_stops if c not in ['vessel']]]
        appended_data.append(df_stops)

    df = pd.concat(appended_data)
    # df.to_excel("json_to_excel.xlsx")

    unplanned = input['unplannedOrders']
    n_unplanned = len(unplanned)
    n_planned = df['discharging20'].sum() + df['discharging40'].sum() + df['discharging45'].sum()

    vessels = df['vessel'].unique()
    unused = []
    for vessel in vessels:
        for s in reversed(range(len(df[df['vessel']==vessel]))):
            if (df[df['vessel']==vessel]['fixedStop'].iloc[s]==False):
                if(df[df['vessel']==vessel]['loading20'].iloc[s]>0 or df[df['vessel']==vessel]['loading40'].iloc[s]>0 or df[df['vessel']==vessel]['loading45'].iloc[s]>0):
                    break
                else:
                    if (df[df['vessel']==vessel]['fixedStop'].iloc[s-1]==True):
                        unused.append(vessel)
            elif (s == len(df[df['vessel']==vessel])-1):
                unused.append(vessel)

    n_stops = 0
    for r in range(1,len(df)):
        vessel1 = df['vessel'].iloc[r-1]
        vessel2 = df['vessel'].iloc[r]
        if (vessel1 == vessel2):
            terminal1 = df['terminalId'].iloc[r-1]
            terminal2 = df['terminalId'].iloc[r]
            if (terminal1 != terminal2):
                n_stops = n_stops + 1
    n_stops = n_stops + len(vessels)

    distance_sailed = 0
    for i in range(len(input['routes'])):
        distance_sailed = distance_sailed + input['routes'][i]['distanceSailed']

    planning_time = end - start

    return n_planned, n_unplanned, unused, n_stops, distance_sailed, planning_time, unplanned

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
bootstrap_stylesheets = ['https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=bootstrap_stylesheets)
server = app.server

def create_planning_board(df):
    global timestamp
    vessels = df.vessel.unique().tolist()
    fig = px.timeline(df, x_start='startTime', x_end='departureTime', y='vessel', color='terminalId', 
                hover_data=["loading20", "loading40", "loading45", "discharging20", "discharging40", "discharging45", "teuOnBoardAfterStop", "weightOnBoardAfterStop"], 
                color_discrete_sequence=px.colors.qualitative.Alphabet, height=len(vessels)*80, width=len(vessels)*400)
    fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
    
    date_format = '%Y-%m-%dT%H:%M:%SZ'
    timestamp = pd.to_datetime(timestamp, format=date_format) + datetime.timedelta(days=2)

    fig.add_vline(x=timestamp)
    # fig.update_traces(marker_line_color='#2E3440', marker_line_width=2)
    # fig.update_layout(font_family='Arial', font_size=14)
    # fig.update_layout(legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=1.02,
    #     xanchor="right",
    #     x=0.6,
    #     font_size=10
    # ))
    return fig

fig = go.Figure(go.Scatter(x=[], y = []))
fig.update_layout(template = None)
fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)

# Define the app layout
app.layout = html.Div([
    dbc.Row([
        dbc.Col(width=3), # A fake col to keep the title in center no matter what...
        dbc.Col(
            html.H1(children=['PMA', html.Sup('TM'), ' web app by ']),
            width=3,
            style={
                'textAlign': 'right',
            }),
        dbc.Col(
            html.Div(
                html.Img(src=dash.get_asset_url('cofano.png')),
                style={'float': 'left'}
            ),
            width=3,
        )
    ], justify='start'),
    dbc.Row([
        dbc.Col(
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Planning Input')
                ]),
                style={
                    'width': '99vw',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'borderColor': '#DA680F',
                    'font-weight': 'bold',
                    'textAlign': 'center',
                    'margin': '10px',
                    'cursor': 'pointer',
                    'color': '#DA680F',
                    'background-color': '#0d0c0b',
                },
                multiple=False
            ),
        ),
    ]),
    html.Div(id='barge_list'),
    html.Br(),
    html.Div(children=[
        html.Label('Travel distance (minimization priority)'),
        dcc.Slider(
            id = 'w1_slider',
            min=0,
            max=1,
            step = 0.1,
            marks={str(i/10) : str(i/10) for i in range(11)},
            value=0.1,
        ),
        html.Label('Number of stops (minimization priority)'),
        dcc.Slider(
            id = 'w2_slider',
            min=0,
            max=1,
            step = 0.1,
            marks={str(i/10) : str(i/10) for i in range(11)},
            value=0.5,
        ),
        html.Label('Unplanned/late containers (minimization priority)'),
        dcc.Slider(
            id = 'w3_slider',
            min=0,
            max=1,
            step = 0.1,
            marks={str(i/10) : str(i/10) for i in range(11)},
            value=0.5,
        ),
    ]),
    html.Br(),
    html.Div([
        dbc.Button('Plan', size='lg', id='make-planning', n_clicks=0, style={
                    'borderColor': '#DA680F',
                    'color': '#DA680F',
                    'background-color': '#0d0c0b',
                    'font-weight': 'bold'}),
    ], style={'textAlign': 'center'}),
    html.Br(),
    html.Div(id='result'),
    html.Br(),
    html.H3(children='Planning KPIs', style={'textAlign': 'center'}),
    html.Div(id='kpis'),
    html.Br(),
    html.H2(children='Planning board', style={'textAlign': 'center'}),
    html.Div([
        html.Div(
            dcc.Graph(
                id = 'planning_board',
                figure = fig,),
        )
    ], style={'overflowX': 'scroll', 
              'height': '85vh',
              'width': '99vw'}),
    html.Br(),
    html.Div(id='output-data-upload'),
    html.Br(),
    html.Div(id='input-data'),
    html.Br(),
    html.Div(id='hoi')
])


# Global variable to store the imported JSON output of PMA
pma_json = None

def parse_contents():
    k = 0
    while (imported_output_json is None):
        k = k + 1

    planning = copy.deepcopy(imported_output_json['routes'])
    appended_data = []
    for v in range(len(planning)):
        stops = planning[v]['stops']
        vessel = planning[v]['vessel']
        df_stops = pd.DataFrame.from_dict(stops)
        df_stops = df_stops.drop(columns=['loadOrders', 'dischargeOrders'])
        df_stops['vessel'] = vessel
        df_stops = df_stops[['vessel'] + [c for c in df_stops if c not in ['vessel']]]
        appended_data.append(df_stops)

    df = pd.concat(appended_data)

    return html.Div([
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                # all three widths are needed
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'whiteSpace': 'normal'
            },
            style_header={
                'backgroundColor': 'orange',
                'fontWeight': 'bold'
            },
            export_format='xlsx',
            export_headers='display',
            # style_cell_conditional=[
            #     {
            #         'if': {'column_id': 'vessel'},
            #         'width': '250px'
            #     },
            # ],
        ),
    ])

orders_df = None
terminals_df = None
vessels_df = None
timestamp = None

def parse_input(contents):
    global orders_df
    global terminals_df
    global vessels_df
    global timestamp

    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        input = json.loads(decoded)

    timestamp = input['timestamp']

    orders = input['orders']
    orders_df = pd.DataFrame.from_dict(orders)
    for i in range(len(orders_df)):
        earliestPickUp = orders_df['loadTimeWindow'].iloc[i]['startDateTime']
        latestPickUp = orders_df['loadTimeWindow'].iloc[i]['endDateTime']
        earliestDeadline = orders_df['dischargeTimeWindow'].iloc[i]['startDateTime']
        latestDeadline = orders_df['dischargeTimeWindow'].iloc[i]['endDateTime']
        if (i > 0):
            orders_df['earliestPickUp'].iloc[i] = earliestPickUp
            orders_df['latestPickUp'].iloc[i] = latestPickUp
            orders_df['earliestDeadline'].iloc[i] = earliestDeadline
            orders_df['latestDeadline'].iloc[i] = latestDeadline
        else:
            orders_df['earliestPickUp'] = earliestPickUp
            orders_df['latestPickUp'] = latestPickUp
            orders_df['earliestDeadline'] = earliestDeadline
            orders_df['latestDeadline'] = latestDeadline

    orders_df = orders_df.drop(columns=['loadExternalId', 'dischargeExternalId', 'loadTimeWindow', 'dischargeTimeWindow', 'loadstopId', 'unloadstopId', 'fine'])
    orders_df = orders_df.drop(columns=['containerNumber', 'bookingIdentifier'])

    terminals = input['terminals']
    terminals_df = pd.DataFrame.from_dict(terminals)
    # for i in range(len(terminals_df)):
    #     latitude = terminals_df['position'].iloc[i]['latitude']
    #     longitude = terminals_df['position'].iloc[i]['longitude']

    #     if (i > 0):
    #         terminals_df['latitude'].iloc[i] = latitude
    #         terminals_df['longitude'].iloc[i] = longitude
    #     else:
    #         terminals_df['latitude'] = latitude
    #         terminals_df['longitude'] = longitude

    #     for d in range(len(terminals_df['openingTimes'][i])):
    #         day = terminals_df['openingTimes'][i][d]['weekDay']
    #         startTime = terminals_df['openingTimes'][i][d]['startTime']
    #         flexStartTime = terminals_df['openingTimes'][i][d]['flexStartTime']
    #         endTime = terminals_df['openingTimes'][i][d]['endTime']
    #         flexEndTime = terminals_df['openingTimes'][i][d]['flexEndTime']
    #         if (i > 0):
    #             terminals_df[day.lower() + 'startTime'].iloc[i] = startTime
    #             terminals_df[day.lower() + 'flexStartTime'].iloc[i] = flexStartTime
    #             terminals_df[day.lower() + 'endTime'].iloc[i] = endTime
    #             terminals_df[day.lower() + 'flexEndTime'].iloc[i] = flexEndTime
    #         else:
    #             terminals_df[day.lower() + 'startTime'] = startTime
    #             terminals_df[day.lower() + 'flexStartTime'] = flexStartTime
    #             terminals_df[day.lower() + 'endTime'] = endTime
    #             terminals_df[day.lower() + 'flexEndTime'] = flexEndTime

    terminals_df = terminals_df.drop(columns=['externalId', 'handlingTime', 'baseStopTime', 'openingTimes', 'callCost', 'callSizeFine', 'position'])

    vessels = input['vessels']
    appended_data2 = []
    for v in range(len(vessels)):
        stops = vessels[v]['stops']
        vessel = vessels[v]['id']
        capacityTEU = vessels[v]['capacityTEU']
        capacityWeight = vessels[v]['capacityWeight']
        capacityReefer = vessels[v]['capacityReefer']
        capacityDangerGoods = vessels[v]['capacityDangerGoods']
        activeTimes = vessels[v]['activeTimes']
        forbiddenTerminals = vessels[v]['forbiddenTerminals']
        df_stops = pd.DataFrame.from_dict(stops)
        df_stops['capacityTEU'] = capacityTEU
        df_stops['capacityWeight'] = capacityWeight
        for i in range(len(df_stops)):
            # df_stops['loadOrders'][i] = ', '.join(df_stops['loadOrders'][i])
            # df_stops['dischargeOrders'][i] = ', '.join(df_stops['dischargeOrders'][i])
            # df_stops.__getitem__('loadOrders').__setitem__(i, len(df_stops['loadOrders'][i]))
            # df_stops.__getitem__('dischargeOrders').__setitem__(i, len(df_stops['dischargeOrders'][i]))
            df_stops['loadOrders'][i] = len(df_stops['loadOrders'][i])
            df_stops['dischargeOrders'][i] = len(df_stops['dischargeOrders'][i])
            # startDateTime = df_stops.loc[:, ('timeWindow', i, 'startDateTime')]
            # endDateTime = df_stops.loc[:, ('timeWindow', i, 'endDateTime')]
            startDateTime = df_stops['timeWindow'][i]['startDateTime']
            endDateTime = df_stops['timeWindow'][i]['endDateTime']
            if (i > 0):
                df_stops['arrivalTime'].iloc[i] = startDateTime
                df_stops['departureTime'].iloc[i] = endDateTime
            else:
                df_stops['arrivalTime'] = startDateTime
                df_stops['departureTime'] = endDateTime
            # for d in range(len(activeTimes)):
            #     day = activeTimes[d]['weekDay'].lower()
            #     startTime = activeTimes[d]['startTime']
            #     endTime = activeTimes[d]['endTime']
            #     if (i > 0):
            #         df_stops[day.lower() + 'StartTime'].iloc[i] = startTime
            #         df_stops[day.lower() + 'EndTime'].iloc[i] = endTime
            #     else:
            #         df_stops[day.lower() + 'StartTime'] = startTime
            #         df_stops[day.lower() + 'EndTime'] = endTime

        df_stops = df_stops.drop(columns=['linestopId', 'timeWindow'])
        df_stops = df_stops[['terminalId', 'arrivalTime', 'departureTime'] + [c for c in df_stops if c not in ['terminalId', 'arrivalTime', 'departureTime']]]
        df_stops['vessel'] = vessel
        # df_stops['capacityReefer'] = capacityReefer
        # df_stops['capacityDangerGoods'] = capacityDangerGoods

        df_stops = df_stops[['vessel'] + [c for c in df_stops if c not in ['vessel']]]
        appended_data2.append(df_stops)

    vessels_df = pd.concat(appended_data2)
    vessels_df = vessels_df.drop(columns=['externalId'])

    return html.Div([
        dash_table.DataTable(
            orders_df.to_dict('records'),
            [{'name': i, 'id': i} for i in orders_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                # all three widths are needed
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'whiteSpace': 'normal'
            },
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
        ),
        html.Br(),
        dash_table.DataTable(
            terminals_df.to_dict('records'),
            [{'name': i, 'id': i} for i in terminals_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                # all three widths are needed
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'whiteSpace': 'normal'
            },
            style_header={
                'backgroundColor': 'green',
                'fontWeight': 'bold'
            },
        ),
        html.Br(),
        dash_table.DataTable(
            vessels_df.to_dict('records'),
            [{'name': i, 'id': i} for i in vessels_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                # all three widths are needed
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'whiteSpace': 'normal'
            },
            style_header={
                'backgroundColor': 'blue',
                'fontWeight': 'bold'
            },
            editable=True
        ),
    ])

# Global variable to store the uploaded JSON data
uploaded_json = None
imported_output_json = None
start = None
end = None

# Function to send JSON data as part of a POST request
def send_json_data(data):
    global start
    url = f'http://localhost:8080/api/planning'
    username = 'cofano'
    password = 'cofano'
    auth = (username, password)
    headers = {'Content-type': 'application/json'}
    r = requests.post(url=url, auth=auth, headers=headers, json=uploaded_json)

    if r.status_code == 200:
        print("POST request was succesful!")
        print("Response data:", r.text)
        start = time.time()
    else:
        print("POST request failed with status code:", r.status_code)
        print("Response data:", r.text)

def adjust_vessels_json(contents, barges_selected):
    df = pd.json_normalize(contents)
    vessels = df['vessels'].values[0]
    fleet = []
    for vessel in vessels:
        fleet.append(vessel['id'])
    not_selected = [x for x in fleet if x not in barges_selected]
    st = set(not_selected)
    ind_not_selected = [i for i, e in enumerate(fleet) if e in st]
    days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
    for i in ind_not_selected:
        contents['vessels'][i]['kilometerCost'] = 100000
        for day in days:
            contents['vessels'][i]['dayCost'][day] = 100000
            contents['vessels'][i]['terminalCallCost'][day] = 100000

    loaded_orders_notfixed = []
    for i in ind_not_selected:
        for j in range(len(contents['vessels'][i]['stops'])):
            if (contents['vessels'][i]['stops'][j]['fixedStop'] == False):
                for k in contents['vessels'][i]['stops'][j]['loadOrders']:
                    loaded_orders_notfixed.append(k)
                contents['vessels'][i]['stops'][j]['loadOrders'] = []
    for i in ind_not_selected:
        for j in range(len(contents['vessels'][i]['stops'])):
            if (contents['vessels'][i]['stops'][j]['fixedStop'] == False):
                res = [k for k in contents['vessels'][i]['stops'][j]['dischargeOrders'] if k not in loaded_orders_notfixed]
                contents['vessels'][i]['stops'][j]['dischargeOrders'] = res
    for i in ind_not_selected:
        for j in range(len(contents['vessels'][i]['stops'])):
            if (contents['vessels'][i]['stops'][j]['fixedStop'] == False):
                if (len(contents['vessels'][i]['stops'][j]['dischargeOrders'])) > 0:
                    contents['vessels'][i]['stops'][j]['fixedStop'] = True


    return contents

def adjust_penalties_json(contents, w1, w2, w3):
    w1 = 10*w1
    w2 = 2*w2
    w3 = 2*w3

    days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
    for i in range(len(contents['vessels'])):
        contents['vessels'][i]['kilometerCost'] = w1*contents['vessels'][i]['kilometerCost']
        for day in days:
            contents['vessels'][i]['terminalCallCost'][day] = w2*contents['vessels'][i]['terminalCallCost'][day]
    for i in range(len(contents['terminals'])):
        contents['terminals'][i]['callCost'] = w2*contents['terminals'][i]['callCost']
    for i in range(len(contents['orders'])):
        contents['orders'][i]['fine'] = w3*contents['orders'][i]['fine']
    return contents

def upload_and_send_json(contents, barges, w1, w2, w3):
    global uploaded_json

    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        uploaded_json = json.loads(decoded)

        uploaded_json = adjust_vessels_json(uploaded_json, barges)
        uploaded_json = adjust_penalties_json(uploaded_json, w1, w2, w3)

        # Send the JSON data as part of a POST request
        response = send_json_data(uploaded_json)
        return [f'Uploaded JSON Data:\n{json.dumps(uploaded_json, indent=4)}\n\nResponse:\n{response}']

    return ['']

def import_output_json(contents):
    global imported_output_json
    global end

    if contents is not None:
        filename = "C:/Users/quiri/Documents/Werk/PlanningJSON.json"
        start_time = time.time()
        k = 0
        file_created = os.path.getmtime(filename)
        while (file_created < start_time):
            if k == 0:
                print('File created:')
                print(os.path.getmtime(filename))
                print('Program executed:')
                print(start_time)
            file_created = os.path.getmtime(filename)
            k = k + 1
        end = time.time()
        f = open(r"C:\Users\quiri\Documents\Werk\PlanningJSON.json", "r")
        input = json.load(f)
        imported_output_json = input
        return [f'Imported PMA output in JSON format']

    return ['']

def adjust_fixed_stop(rows):
    print(rows)
    return rows


@app.callback(
    Output('input-data', 'children'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_input(contents):
    return parse_input(contents)

@app.callback(
    Output('input-data', 'data'),
    Input('input-data', 'data'),
    prevent_initial_call=True
)
def update_columns(rows):
    rows = adjust_fixed_stop(rows)
    return print('hoi')

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    Input('make-planning', 'n_clicks'),
    Input('checklist', 'value'),
    Input('w1_slider', 'value'),
    Input('w2_slider', 'value'),
    Input('w3_slider', 'value'),
    prevent_initial_call=True
)
def update_output(contents, n_clicks, barges, w1, w2, w3):
    if (n_clicks > 0):
        upload_and_send_json(contents, barges, w1, w2, w3)
        import_output_json(contents)
        return parse_contents()

@app.callback(
    Output('planning_board', 'figure'),  
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_figure(contents):    
    if contents is None:
        raise PreventUpdate
    k = 0
    while (imported_output_json is None):
        k = k + 1
    result = process_input_for_planning_board(imported_output_json)
    return create_planning_board(result)

@app.callback(
    Output('kpis', 'children'),  
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def show_kpis(contents):
    if contents is None:
        raise PreventUpdate
    k = 0
    while (imported_output_json is None):
        k = k + 1
    n_planned, n_unplanned, unused, n_stops, distance_sailed, planning_time, unplanned = load_and_process_kpis(imported_output_json)

    if (len(unused) > 1):
        unused = ", ".join(unused)
    else:
        unused = str(unused[0])

    to_print = ['Created a schedule! Number of containers planned: ' + str(n_planned) + ', Number of unplanned containers: ' + 
                str(n_unplanned) + ', Unused barges: ' + str(unused) + ', Total number of stops: ' + 
                str(n_stops) + ', Total distance to sail: ' + str(distance_sailed) + ' KM' + ', Number of seconds to create schedule: ' + str(round(planning_time)) + '. \n']
    if (len(unplanned) > 0):
        unplanned_print = ['\n Unplanned containers: ']
        for u in range(len(unplanned)):
            unplanned_print = unplanned_print + [str(unplanned[u])]
            if (u < len(unplanned) - 1):
                unplanned_print = unplanned_print + [", "]
        to_print = to_print + unplanned_print

    return html.Div(to_print)

@app.callback(
    Output('barge_list', 'children'),  
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def barge_checklist(contents):
    if contents is None:
        raise PreventUpdate
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    uploaded_json = json.loads(decoded)
    barges = get_barges_from_json(uploaded_json)
    return html.Div(children=[
            html.Label('Select vessels to plan'),
            dbc.Checklist(id='checklist', options=barges, value=barges, inline=True, label_checked_style={"color": "#DA680F", 'font-weight': 'bold',},
            input_checked_style={
                "backgroundColor": "#0d0c0b", 
                "borderColor": "#DA680F",
            }, style={'textAlign': 'center'})
            ])

# @app.callback(
#     Output('input-analysis', 'children'),  
#     Input('upload-data', 'contents'),
#     prevent_initial_call=True
# )
# def barge_checklist(contents):
#     if contents is None:
#         raise PreventUpdate
#     print(contents)
#     print(contents)
    

#     return html.Div(children=[
#             html.Label('Select vessels to plan'),
#             dbc.Checklist(id='checklist', options=barges, value=barges, inline=True, label_checked_style={"color": "#DA680F", 'font-weight': 'bold',},
#             input_checked_style={
#                 "backgroundColor": "#0d0c0b", 
#                 "borderColor": "#DA680F",
#             }, style={'textAlign': 'center'})
#             ])

if __name__ == '__main__':
    app.run_server(debug=False)
