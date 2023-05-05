"""
    To run: 
        run 'nbaPassDiagram.py'
        input season and team at terminal
"""
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import math
import numpy as np

def readCsvFile(season: str, team_abbreviation: str): # Read csv file, find target seaon and team
    df = pd.read_csv('Pass_data/Pass_Data_21_22.csv')
    while True:
        if season not in df['SEASON'].values and team_abbreviation not in df['TEAM_ABBREVIATION'].values:
            season = input("Season and team input not found!\nEnter season:") or '2021-22'
            team_abbreviation = input("Enter team(abbreviation):") or 'GSW'
        elif season not in df['SEASON'].values:
            season = input("Season input not found!\nEnter season:") or '2021-22'
        elif team_abbreviation not in df['TEAM_ABBREVIATION'].values:
            team_abbreviation = input("Team input not found!\nEnter team(abbreviation):") or 'GSW'
        else:
            df = df[(df['SEASON'] == season) & (df['TEAM_ABBREVIATION'] == team_abbreviation)]
            if not df.empty:
                return df

def buildPlayerStats(df: pd.DataFrame): # Build players' PASS, FGM, FGA total stats and add total accuracy
    player_names = np.union1d(df['PLAYER_NAME'].unique(), df['PASS_TO'].unique())
    df_total = pd.DataFrame({'PLAYER_NAME': player_names})
    df_total = df_total.merge(df.groupby('PASS_TO')[['FGM', 'FGA']].sum().reset_index().rename(columns={'PASS_TO': 'PLAYER_NAME', 'FGM': 'TOTAL_FGM', 'FGA': 'TOTAL_FGA'}), on='PLAYER_NAME', how='left')
    df_total['TOTAL_ACCURACY'] = np.where(df_total['TOTAL_FGA'] == 0, 0, round(df_total['TOTAL_FGM'] / df_total['TOTAL_FGA'], 2))
    df_total = df_total.merge(df.groupby('PLAYER_NAME')['PASS'].sum().reset_index().rename(columns={'PASS': 'TOTAL_PASS'}), on='PLAYER_NAME', how='left')
    return df_total.fillna(0)

def buildPassList(df: pd.DataFrame, player_total: pd.DataFrame): # Build pass list to show in diagram by filtering PASS > 99 and add passing quality based on player total accuracy
    df_passes = df[['PLAYER_NAME', 'PASS_TO', 'PASS', 'FGM', 'FGA',]][df['PASS'] > 99]
    df_passes = df_passes.merge(player_total[['PLAYER_NAME', 'TOTAL_ACCURACY']], on='PLAYER_NAME', how='left')
    df_passes['QUALITY'] = np.where(df_passes['TOTAL_ACCURACY'] * df_passes['FGA'] == 0, 0, df_passes['FGM'] / df_passes['FGA'] / df_passes['TOTAL_ACCURACY'])
    df_passes = df_passes.drop(['TOTAL_ACCURACY'], axis=1)
    return df_passes

def filterOneWayPasses(df_passes: pd.DataFrame): # Filter pass list that only has one player passing to another
    reversed_pairs = df_passes[['PLAYER_NAME', 'PASS_TO']].apply(lambda x: tuple(sorted(x)), axis=1)
    df_oneWayPasses = df_passes.copy()
    df_oneWayPasses['reversed_pairs'] = reversed_pairs
    df_oneWayPasses = df_oneWayPasses[~df_oneWayPasses.duplicated(subset=['reversed_pairs'], keep=False)]
    return df_oneWayPasses.drop('reversed_pairs', axis=1)
'''
def filterTwoWayPasses(df_passes: pd.DataFrame): # Produces Cartesian Product, not efficient
    index = pd.MultiIndex.from_product([df_passes.index, df_passes.index])
    df1_reindexed = df_passes.reindex(index.get_level_values(0))
    df2_reindexed = df_passes.add_suffix('_2').reindex(index.get_level_values(1))
    merged = pd.concat([df1_reindexed.reset_index(drop=True), df2_reindexed.reset_index(drop=True)], axis=1)
    df_twoWayPasses = merged[(merged['PLAYER_NAME'] == merged['PASS_TO_2']) & (merged['PASS_TO'] == merged['PLAYER_NAME_2'])]
    df_twoWayPasses['player_pair'] = df_twoWayPasses.apply(lambda row: tuple(sorted([row['PLAYER_NAME'], row['PASS_TO']])), axis=1)
    df_twoWayPasses = df_twoWayPasses.drop_duplicates(subset='player_pair')
    return df_twoWayPasses.drop('player_pair', axis=1)
'''

def filterTwoWayPasses(df_passes: pd.DataFrame): # Filter pass list that both players have passed to each other
    df_twoWayPasses = df_passes
    df_twoWayPasses['player_pair'] = df_twoWayPasses.apply(lambda row: tuple(sorted([row['PLAYER_NAME'], row['PASS_TO']])), axis=1)
    df_twoWayPasses = df_twoWayPasses[df_twoWayPasses.duplicated(subset='player_pair', keep=False)]
    df_reversed = df_twoWayPasses[df_twoWayPasses.duplicated(subset='player_pair', keep='first')]
    df_twoWayPasses = df_twoWayPasses.drop_duplicates(subset='player_pair', keep='first')
    df_twoWayPasses = df_twoWayPasses.merge(df_reversed, on = 'player_pair', suffixes=('', '_2'))
    df_twoWayPasses = df_twoWayPasses.drop('player_pair', axis=1)
    return df_twoWayPasses


def createNodeTraces(G: nx.Graph, pos: dict, player_total: pd.DataFrame): # Create node traces
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        text=list(player_total['PLAYER_NAME']),
        hovertext = ['Name: ' + pn + 
                '_Total FGM: ' + str(pfa) + 
                '_Total FGA: ' + str(pfm) + 
                '_Total Accuracy: ' + str(pa)  +
                '_Total Passes: ' + str(pp)
                for pn, pfa, pfm, pa, pp in zip(player_total['PLAYER_NAME'], player_total['TOTAL_FGM'], player_total['TOTAL_FGA'], player_total['TOTAL_ACCURACY'], player_total['TOTAL_PASS'])],
        mode='markers+text',
        textposition='top center',
        marker=dict(size=10, color='blue')
    )
    return node_trace

#0-2 -> 0.01-0.03
def calcMidIncr(srcX, srcY, tarX, tarY): # calculate the shifting parameters for two way pass when drawing diagram
    incr_size = 1 + math.sqrt((tarX - srcX)**2 + (tarY - srcY)**2) / 2 * 2
    slope = y = x = 0
    if srcX == tarX:
        return (incr_size/100, 0)
    elif srcY == tarY:
        return (0, incr_size/100)
    else:
        slope = -1 / ((tarY-srcY) / (tarX-srcX))
        y = (slope / math.sqrt(slope ** 2 + 1)) * incr_size
        x = y / slope
    return (x/100, y/100)

def add_arrow(source_x: float, target_x: float, source_y: float, target_y: float,
               arrowLength=0.025, arrowAngle=30): # Arrow adding for edges(referenced from github: https://github.com/redransil/plotly-dirgraph)
    pointx = source_x + (target_x - source_x) / 2
    pointy = source_y + (target_y - source_y) / 2
    etas = math.degrees(math.atan((target_x - source_x) / (target_y - source_y)))
    
    signx = (target_x - source_x) / abs(target_x - source_x)
    signy = (target_y - source_y) / abs(target_y - source_y)

    dx = arrowLength * math.sin(math.radians(etas + arrowAngle))
    dy = arrowLength * math.cos(math.radians(etas + arrowAngle))
    none_spacer = None
    arrow_line_x = pointx - signx ** 2 * signy * dx
    arrow_line_y = pointy - signx ** 2 * signy * dy

    arrow_line_1x_coords = [pointx, arrow_line_x, none_spacer]
    arrow_line_1y_coords = [pointy, arrow_line_y, none_spacer]

    dx = arrowLength * math.sin(math.radians(etas - arrowAngle))
    dy = arrowLength * math.cos(math.radians(etas - arrowAngle))
    none_spacer = None
    arrow_line_x = pointx - signx ** 2 * signy * dx
    arrow_line_y = pointy - signx ** 2 * signy * dy

    arrow_line_2x_coords = [pointx, arrow_line_x, none_spacer]
    arrow_line_2y_coords = [pointy, arrow_line_y, none_spacer]

    x_arrows = arrow_line_1x_coords + arrow_line_2x_coords
    y_arrows = arrow_line_1y_coords + arrow_line_2y_coords

    return x_arrows, y_arrows

def createOneWayEdges(qmin: int, mnrange: int, df_oneWayPasses: pd.DataFrame, pos: dict): # Create and draw edge traces for one way passes
    global fig
    for idx, row in df_oneWayPasses.iterrows():
        color_val = (row['QUALITY'] - qmin) / mnrange
        color_tuple = (color_val * 255, 0, (1 - color_val) * 255)
        fig.add_trace( # first half of the edge
            go.Scatter(
                x = [pos[row['PLAYER_NAME']][0], (pos[row['PLAYER_NAME']][0]+pos[row['PASS_TO']][0])/2, None],
                y = [pos[row['PLAYER_NAME']][1], (pos[row['PLAYER_NAME']][1]+pos[row['PASS_TO']][1])/2, None],
                line = dict(width = row['PASS'] * 0.003, color = f'rgb{color_tuple}'),
                text = f"Pass: {row['PASS']} \nFGM: {row['FGM']} \nFGA: {row['FGA']} \nQuality: {row['QUALITY']:.2f}",
                hoverinfo='text',
                mode='lines',
            )
        )
        arrowPos = add_arrow(pos[row['PLAYER_NAME']][0], 
                            pos[row['PASS_TO']][0], 
                            pos[row['PLAYER_NAME']][1], 
                            pos[row['PASS_TO']][1])
        fig.add_trace( # add arrow
            go.Scatter(
                x = arrowPos[0],
                y = arrowPos[1],
                line = dict(width = row['PASS'] * 0.003, color = f'rgb{color_tuple}'), 
                hoverinfo='none', 
                mode='lines'
            )
        )
        fig.add_trace( # second half of the edge
            go.Scatter(
                x = [(pos[row['PLAYER_NAME']][0]+pos[row['PASS_TO']][0])/2, pos[row['PASS_TO']][0], None],
                y = [(pos[row['PLAYER_NAME']][1]+pos[row['PASS_TO']][1])/2, pos[row['PASS_TO']][1], None],
                line = dict(width = row['PASS'] * 0.003, color = f'rgb{color_tuple}'),
                text = f"Pass: {row['PASS']} \nFGM: {row['FGM']} \nFGA: {row['FGA']} \nQuality: {row['QUALITY']:.2f}",
                hoverinfo='text',
                mode='lines',
            )
        )

def createTwoWayEdges(qmin: int, mnrange: int, df_twoWayPasses: pd.DataFrame, pos: dict): # Create and draw edge traces for two way passes
    global fig
    for idx, row in df_twoWayPasses.iterrows(): # One of the edge in the two way passes
        color_val = (row['QUALITY'] - qmin) / mnrange
        color_tuple = (color_val * 255, 0, (1 - color_val) * 255)
        vertic = calcMidIncr(pos[row['PLAYER_NAME']][0], pos[row['PLAYER_NAME']][1], pos[row['PASS_TO']][0], pos[row['PASS_TO']][1])
        fig.add_trace( # first half of the edge
            go.Scatter(
                x = [pos[row['PLAYER_NAME']][0], (pos[row['PASS_TO']][0]+pos[row['PLAYER_NAME']][0])/2 + vertic[0], None],
                y = [pos[row['PLAYER_NAME']][1], (pos[row['PASS_TO']][1]+pos[row['PLAYER_NAME']][1])/2 + vertic[1], None],
                line = dict(width = row['PASS'] * 0.003, color = f'rgb{color_tuple}'),
                text = f"Pass: {row['PASS']} \nFGM: {row['FGM']} \nFGA: {row['FGA']} \nQuality: {row['QUALITY']:.2f}",
                hoverinfo='text',
                mode='lines',
            )
        )
        arrowPos = add_arrow(pos[row['PLAYER_NAME']][0], 
                            pos[row['PASS_TO']][0]+vertic[0]*2, 
                            pos[row['PLAYER_NAME']][1], 
                            pos[row['PASS_TO']][1]+vertic[1]*2)
        fig.add_trace( # add arrow
            go.Scatter(
                x = arrowPos[0],
                y = arrowPos[1],
                line = dict(width = row['PASS'] * 0.003, color = f'rgb{color_tuple}'), 
                hoverinfo='none', 
                mode='lines'
            )
        )
        fig.add_trace( # second half of the edge
            go.Scatter(
                x = [(pos[row['PASS_TO']][0]+pos[row['PLAYER_NAME']][0])/2 + vertic[0], pos[row['PASS_TO']][0], None],
                y = [(pos[row['PASS_TO']][1]+pos[row['PLAYER_NAME']][1])/2 + vertic[1], pos[row['PASS_TO']][1], None],
                line = dict(width = row['PASS'] * 0.003, color = f'rgb{color_tuple}'),
                text = f"Pass: {row['PASS']} \nFGM: {row['FGM']} \nFGA: {row['FGA']} \nQuality: {row['QUALITY']:.2f}",
                hoverinfo='text',
                mode='lines',
                marker= dict(size=10,symbol= "arrow-bar-up")
            )
        )
    for idx, row in df_twoWayPasses.iterrows(): # The other edge
        color_val = (row['QUALITY_2'] - qmin) / mnrange
        color_tuple = (color_val * 255, 0, (1 - color_val) * 255)
        vertic = calcMidIncr(pos[row['PLAYER_NAME_2']][0], pos[row['PLAYER_NAME_2']][1], pos[row['PASS_TO_2']][0], pos[row['PASS_TO_2']][1])
        fig.add_trace( # first half of the edge
            go.Scatter(
                x = [pos[row['PLAYER_NAME_2']][0], (pos[row['PASS_TO_2']][0]+pos[row['PLAYER_NAME_2']][0])/2 - vertic[0], None],
                y = [pos[row['PLAYER_NAME_2']][1], (pos[row['PASS_TO_2']][1]+pos[row['PLAYER_NAME_2']][1])/2 - vertic[1], None],
                line = dict(width = row['PASS_2'] * 0.003, color = f'rgb{color_tuple}'),
                text = f"Pass: {row['PASS_2']} \nFGM: {row['FGM_2']} \nFGA: {row['FGA_2']} \nQuality: {row['QUALITY_2']:.2f}",
                hoverinfo='text',
                mode='lines',
            )
        )
        arrowPos = add_arrow(pos[row['PLAYER_NAME_2']][0], 
                            pos[row['PASS_TO_2']][0]-vertic[0]*2, 
                            pos[row['PLAYER_NAME_2']][1], 
                            pos[row['PASS_TO_2']][1]-vertic[1]*2)
        fig.add_trace( # add arrow
            go.Scatter(
                x = arrowPos[0],
                y = arrowPos[1],
                line = dict(width = row['PASS'] * 0.003, color = f'rgb{color_tuple}'), 
                hoverinfo='none', 
                mode='lines'
            )
        )
        fig.add_trace( # second half of the edge
            go.Scatter(
                x = [(pos[row['PASS_TO_2']][0]+pos[row['PLAYER_NAME_2']][0])/2 - vertic[0], pos[row['PASS_TO_2']][0], None],
                y = [(pos[row['PASS_TO_2']][1]+pos[row['PLAYER_NAME_2']][1])/2 - vertic[1], pos[row['PASS_TO_2']][1], None],
                line = dict(width = row['PASS_2'] * 0.003, color = f'rgb{color_tuple}'),
                text = f"Pass: {row['PASS_2']} \nFGM: {row['FGM_2']} \nFGA: {row['FGA_2']} \nQuality: {row['QUALITY_2']:.2f}",
                hoverinfo='text',
                mode='lines',
            )
        )

# Main()
season = input("Enter season:") or '2021-22'
team_abbrev = input("Enter team(abbreviation):") or 'GSW'
df = readCsvFile(season, team_abbrev)
df_total = buildPlayerStats(df)
#print(df_total)

df_passes = buildPassList(df, df_total)
#print(df_passes)
df_oneWayPasses = filterOneWayPasses(df_passes)
#print(df_oneWayPasses)
df_twoWayPasses = filterTwoWayPasses(df_passes)
#print(df_twoWayPasses)

G = nx.Graph()
G.add_nodes_from(df_total['PLAYER_NAME'])
pos = nx.circular_layout(list(df_total['PLAYER_NAME']))
fig = go.Figure()
fig.add_trace(createNodeTraces(G, pos, df_total))

qmin = df_passes['QUALITY'].min()
qmax = df_passes['QUALITY'].max()
mnrange = qmax - qmin
createOneWayEdges(qmin, mnrange, df_oneWayPasses, pos)
createTwoWayEdges(qmin, mnrange, df_twoWayPasses, pos)

fig.update_layout(yaxis = dict(scaleanchor = "x", scaleratio = 1), plot_bgcolor='rgb(255,255,255)', showlegend=False)
fig.show()