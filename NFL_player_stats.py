from numpy.lib.function_base import extract
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from os import listdir
from os.path import isfile, join
import re

# Pandas formatting being set to float 2 decimal place for readability.
pd.set_option('display.float_format', lambda x: '%.02f' % x)

'''NFL_player_stats.py

Author: Greg Slowski, ENSF592, UCalgary
Group 15

This file collects and uses NFL player statistics from the 2019 and 2020 NFL season.
Files are collected as excel (.xlsx) files within the PFR Data folder of the directory.
Passing, rushing, and receiving and receiving yards have been exported from https://www.pro-football-reference.com

This file requires pandas, numpy, matplotlib, and seaborn to be installed within the Python environment you are running this file on.

'''


def import_data(file_name):
    '''
    The purpose of this function is to extract the player stats from PFR Data into usable arrays.

        Parameters:
            filename (str) : the filename of the dataset located in PFR Data Totals

        Returns:
            PFR_data (pandas.core.frame.DataFrame): positional and year specific data array adjusted to be multi-indexed

    '''
    # Importing the current dataset to an array using pandas
    data = pd.read_excel('PFR Data Totals/'+file_name)

    # Using regex to correct player names that previously noted if players achieved certain accolades denoted with * and +, and title their names.
    data['Player'] = data['Player'].str.replace('(\s[\*\+]+|[\*\+]+)', '') #([[A-Za-z.\'-]+[/s[A-Za-z.\'-]]+])
    data['Player'] = data['Player'].str.title()

    # Using regex to extract the 2 letter identifier for position, ie. RB: runningback, QB: quarterback etc., then upper casing them.
    data['Pos'] = data['Pos'].str.extract('([A-Za-z][A-Za-z])', expand=False)
    data['Pos'] = data['Pos'].str.upper()

    # Creating index slices for the six indexes to be used.
    player_idx = data.loc[pd.IndexSlice[:],'Player']
    position_idx = data.loc[pd.IndexSlice[:],'Pos']
    team_idx = data.loc[pd.IndexSlice[:],'Tm']
    age_idx = data.loc[pd.IndexSlice[:],'Age']
    games_idx = data.loc[pd.IndexSlice[:],'G']
    games_started_idx = data.loc[pd.IndexSlice[:],'GS']

    # Columns to be made index's are dropped - as well as rank which is irrelevant once merged and resorted.
    data = data.drop(['Rk', 'Player', 'Tm', 'Age', 'Pos', 'G', 'GS'], axis='columns')

    # Creating a multi-index for rows to include season, season found by a regex in the file name.
    year = re.compile('[0-9]+')
    year = year.findall(file_name)[0]
    row_idxs = [player_idx , [year for i in range(len(player_idx))], position_idx, team_idx, age_idx, games_idx, games_started_idx]
    row_midxs = pd.MultiIndex.from_arrays(row_idxs, names=['Player', 'Season', 'Pos', 'Team', 'Age', 'Games', 'Started'])

    # Converting data into a Pandas data frame using the noted multiindex and column names from the original data
    PFR_data = pd.DataFrame(data.values, index = row_midxs, columns=data.columns)

    # Cleaned data is returned.
    return PFR_data


def merge_and_clean_data(PFR_data):
    '''
    The purpose of this function is to merge and join the player stats from multiple PFR Data sheets into one large usable array for all data.

        Parameters:
            PFR_data (list): list of multiple (in this case 6) data arrays from importing the individual datasets.

        Returns:
            PFR_data_all (pandas.core.frame.DataFrame) : data array containing all information merged and cleaned from the multiple data sets we imported.

    '''

    # Merging 6 arrays by index, retaining all values, in order of 2019 passing, receiving, then rushing. This is repeated for 2020.
    PFR_data_all_even = pd.merge(PFR_data[0], PFR_data[2], left_index=True, right_index=True, how='outer')
    for i in [4, 1, 3, 5]:
        
        # The duplicate column names are suffixed with "_delete", ie. the 2020 columns.
        PFR_data_all_even = pd.merge(PFR_data_all_even, PFR_data[i], left_index=True, right_index=True, how='outer',  suffixes=["", "_delete"])

    # Merging 6 arrays by index, retaining all values, in order of 2020 passing, receiving, then rushing. This is repeated for 2019.
    PFR_data_all_odd = pd.merge(PFR_data[1], PFR_data[3], left_index=True, right_index=True, how='outer')
    for i in [5, 0, 2, 4]:

        # The duplicate column names are suffixed with "_delete", ie. the 2019 columns.
        PFR_data_all_odd = pd.merge(PFR_data_all_odd, PFR_data[i], left_index=True, right_index=True, how='outer',  suffixes=["", "_delete"])

    # Using the overlap (combine_first) of the previous two merged dataframes, the data is overlayed so redundant columns in the second half can be deleted,
    # while blank spaces are filled.
    PFR_data_all = PFR_data_all_even.combine_first(PFR_data_all_odd)

    # Using a mask to identify which columns have been marked to delete, returns the dataframe of all columns not containing delete in the name.
    PFR_data_all = PFR_data_all[PFR_data_all.columns[~PFR_data_all.columns.str.contains('delete')]]
    
    # Receiving fumbles are duplicate data of rushing fumbles, they are dropped. QB record as the data is not relevant to individual player stats.
    PFR_data_all = PFR_data_all.drop(['ReFmb', 'QBrec'], axis=1)

    # Adding stat categories to columns as a multi index
    # Column category index is initialized to the length of stats available, correlating to passing, receiving, and rushing.
    category_idx = []
    for i in range(23):
        category_idx.append('Passing')
    for i in range(11):
        category_idx.append('Receiving')
    for i in range(8):
        category_idx.append('Rushing')

    # Column multi index is created and applied using current column names and the category index created above.
    col_idx = pd.MultiIndex.from_arrays([category_idx, PFR_data_all.columns], names=['Category', 'Stat'])
    PFR_data_all.columns = col_idx

    # Zero filling NaN spaces, converting all array elements to floats
    PFR_data_all = PFR_data_all.fillna(0).astype(float)

    # Catch percentage needs to be adjusted, converting to float changed numbers to decimal representations.
    PFR_data_all['Receiving', 'Ctch%'] = PFR_data_all['Receiving', 'Ctch%']*100

    # Merged, cleaned dataset is returned.
    return PFR_data_all


def fantasy_calc(PFR_data_all, ppr_scoring):
    '''
    The purpose of this function is to calculate fantasy relevant columns and append them to the input array.

        Parameters:
            PFR_data_all (pandas.core.frame.DataFrame) : data array containing all merged and cleaned data.
            ppr_scoring (float) : point per reception value entered by the user, such as 0, 0.5 or 1.

        Returns:
            PFR_data_all (pandas.core.frame.DataFrame) : sorted data array containing all merged and cleaned data, with appended fantasy stat columns.

    '''

    # Fantasy points are calculated using the standard scoring approach, and the input point per reception by the user, then appended to the data array.
    PFR_data_all['Fantasy', 'FPts'] = 4 * PFR_data_all['Passing', 'PaTD'] + .04 * PFR_data_all['Passing', 'PaYds'] - 2 * PFR_data_all['Passing', 'Int'] \
                                        + ppr_scoring * PFR_data_all['Receiving', 'Rec'] + 0.1 * PFR_data_all['Receiving', 'ReYds'] + 6 * PFR_data_all['Receiving', 'ReTD'] \
                                            + 0.1 * PFR_data_all['Rushing', 'RuYds'] + 6 * PFR_data_all['Rushing', 'RuTD'] - 2 * PFR_data_all['Rushing', 'RuFmb']

    # Fantasy points per game, player usage (sometimes called volume), and total touchdowns are calculated and appended to the data array.
    PFR_data_all['Fantasy', 'FPts/G'] = PFR_data_all['Fantasy', 'FPts'] / PFR_data_all.index.to_frame()['Games']
    PFR_data_all['Fantasy', 'Usage'] = PFR_data_all['Receiving', 'Tgt'] + PFR_data_all['Rushing', 'RuAtt'] + PFR_data_all['Passing', 'PaAtt']
    PFR_data_all['Fantasy', 'TDs'] = PFR_data_all['Passing','PaTD'] + PFR_data_all['Receiving','ReTD'] + PFR_data_all['Rushing','RuTD']

    # The data array is sorted based upon fantasy points per game so that the most notable players are at the top of the list.
    PFR_data_all = PFR_data_all.sort_values(('Fantasy', 'FPts/G'), ascending=False)

    # The sorted and appended data array is returned.
    return PFR_data_all


def get_user_input(PFR_data_all):
    '''
    The purpose of this function is to receive user input to understand what outputs they would like to see, and gather the pertinent 
    variables to calculating some of the appended stats based on how they play fantasy football.

        Parameters:
            PFR_data_all (pandas.core.frame.DataFrame) : data array containing all merged and cleaned data.

        Returns:
            ppr_selection (float) :point per reception value entered by the user, such as 0, 0.5 or 1.
            team_amount_selection (int) : number of teams in the users fantasy football league.
            season_selection (String) : user choice of either the 2019 or 2020 season or both "2019-2020".
            index_selection (String) : user choice of either player or position to view.
            player_selection (String) : user choice of player if player index is chosen.
            position_selection (String) : user choice of position if position index is chosen.

    '''

    # User selection for ppr scoring format.
    while True:
        # Try except will only accept a numerical input of 0 or greater. Otherwise reprompted.
        try: 
            ppr_selection = float(input("Please enter the fantasy points-per-reception (ppr) scoring format, ie. 0, 0.5, 1: "))
            if ppr_selection >= 0:
                break
            else:
                raise ValueError
        except ValueError:
            print("Please enter a valid scoring format greater than 0.")

        # User is prompted to enter the number of teams in their league.
    while True:
        # Try except will only accept integers between 2 and 20 as fantasy football wouldnt really make any sense outside of these bounds. Otherwise reprompted.
        try: 
            team_amount_selection = int(input("Please enter number of league teams between 2 and 20, typically 10 or 12: "))
            if (team_amount_selection >= 2 and team_amount_selection <= 20):
                break
            else:
                raise ValueError
        except ValueError:
            print("Please enter a valid number of teams.")

    # User selection for which seasons they would like to get player or position data from.
    while True:
        # Try except will only accept 2019, 2020, or "2019-2020". Otherwise reprompted.
        try: 
            season_selection = input("Which of the two available season(s) for player or position would you like to have tabulated? ie. 2019, 2020, or 2019-2020: ")
            if season_selection == '2019' or season_selection == '2020' or season_selection == '2019-2020':
                break
            else:
                raise ValueError
        except ValueError:
            print("Please enter one of the two seasons available or '2019-2020'.")

    # player, position, and number of team entries are initialized to n/a so the output does not fail depending on decision tree direction taken below.
    player_selection = 'n/a'
    position_selection = 'n/a'
    
    # User selection for fantasy stats of individual player or position stat output.
    while True:
        # Try except will only accept 1 (individual player) or 2 (position) input. Otherwise reprompted.
        try: 
            stat_selection = int(input("Would you like to see individual player stats (enter 1), or player stats by position (enter 2)? "))

            # If user selects 1 (individual player), they are prompted to enter the player name they would like to get stats for.
            if stat_selection == 1:
                index_selection = 'Player'
                while True:
                    # Try except will only accept player names that exist in the data's 'Player' index. Otherwise reprompted.
                    try:
                        player_selection = input("Please enter the full player name, ie. Derrick Henry, Aaron Rodgers: ")
                        if np.any(PFR_data_all.index.to_frame()['Player'] == player_selection):
                            break
                        else:
                            raise ValueError
                    except ValueError:
                        print("Please enter a valid player.")
                break
            # If user selects 2 (position), they are prompted to enter the position they would like to get stats for.
            elif stat_selection == 2:
                index_selection = 'Pos'
                while True:
                    # Try except will only accept valid positions that exist in the data's 'Pos' index. Otherwise reprompted.
                    try:
                        position_selection = input("Please enter the position, ie. QB, RB, WR, TE: ")
                        if np.any(PFR_data_all.index.to_frame()['Pos'] == position_selection):
                            break
                        else:
                            raise ValueError
                    except ValueError:
                        print("Please enter a valid position.")
                break
            else:
                raise ValueError
        except ValueError:
            print("Please enter 1 or 2.")
    
    # All user selections are returned.
    return ppr_selection, team_amount_selection, season_selection, index_selection, player_selection, position_selection


def convert_df_to_fantasy(PFR_data_pos):
    '''
    The purpose of this function is to convert a large player dataset to a fantasy relevant dataset, including primarily only metrics
    that produce fantasy scoring, and associated results.

        Parameters:
            PFR_data_pos (pandas.core.frame.DataFrame) : data array containing all merged and cleaned data, filtered to the user chosen position.

        Returns:
            PFR_data_fantasy (pandas.core.frame.DataFrame) : data array containing only fantasy point calculation relevant data, with value-over-replacement appended.

    '''

    # Creating a new array that only includes the columns that directly effect fantasy scoring.
    PFR_data_fantasy = PFR_data_pos.loc[:, [('Passing', 'PaYds'), ('Passing', 'PaTD'), ('Passing', 'Int'), ('Receiving', 'Rec'), ('Receiving', 'ReYds'), \
                                                                    ('Receiving', 'ReTD'), ('Rushing', 'RuYds'), ('Rushing', 'RuTD'), ('Rushing', 'RuFmb'), \
                                                                        ('Fantasy', 'FPts'), ('Fantasy', 'FPts/G')]]

    # Converting the games played index to a column, dropping the index afterward.
    PFR_data_fantasy['Fantasy', 'Games'] = PFR_data_fantasy.index.to_frame()['Games']
    PFR_data_fantasy = PFR_data_fantasy.reset_index(level='Games', drop=True)

    # Masking to prioritize displaying players with some fantasy football relevance as outside these bounds they likely wouldnt be on fantasy rosters.
    relevant_players = (PFR_data_fantasy['Fantasy', 'FPts/G'] >= 4) & (PFR_data_fantasy['Fantasy', 'Games'] > 2.5)
    PFR_data_fantasy = PFR_data_fantasy[relevant_players]

    # The updated fantasy relevant position specified array is returned.
    return PFR_data_fantasy


def position_trends(PFR_data_all):
    '''
    The purpose of this function is to produce plots indicating fantasy point trends by position for usage (or volume) and touchdowns.

        Parameters:
            PFR_data_all (pandas.core.frame.DataFrame) : data array containing all merged and cleaned data.

        Returns:
            n/a

    '''

    # Pivot table data is tabulated removing indexes and hierarchical ordering of columns for easier pivot table production. Only pertinent data to the pivot tables
    # is brought in after reducing index's.
    PFR_data_all = PFR_data_all.reset_index()
    data_pivot_table = pd.DataFrame({'FPts': PFR_data_all['Fantasy', 'FPts'], 'FPts/G': PFR_data_all['Fantasy', 'FPts/G'], 'Player': PFR_data_all['Player',''], \
                                            'Usage': PFR_data_all['Fantasy','Usage'], 'TDs': PFR_data_all['Fantasy','TDs'], 'Pos': PFR_data_all['Pos',''], \
                                                    'Age': PFR_data_all['Age','']})

    # Pivot table for fantasy points based on usage and position is generated.
    usage_pivot = data_pivot_table.pivot_table('FPts', index='Pos', columns='Usage')

    # Pivot table for fantasy points based on touchdown total and position is generated.
    TD_pivot = data_pivot_table.pivot_table('FPts', index='Pos', columns='TDs')

    # Mask to filter for the 4 main fantasy positions is applied to both pivot tables.
    mask_qb_rb_wr_te = [len({usage_pivot.index.to_frame()['Pos'][i]} & {'QB', 'RB', 'WR', 'TE'}) > 0 for i in range(usage_pivot.shape[0])]
    usage_pivot = usage_pivot[mask_qb_rb_wr_te]
    TD_pivot = TD_pivot[mask_qb_rb_wr_te]

    # 2 subplots are generated from the two pivot tables, figure titled appropriately.
    metric_compare, (top, bottom) = plt.subplots(2, figsize=(8,9))
    metric_compare.suptitle('Usage vs TD value for Fantasy Points in seasons 2019-2020')
    top.plot(usage_pivot.transpose(), marker = '.', lw = 0)
    bottom.plot(TD_pivot.transpose(), marker = '.', lw = 0)

    # Axes labels, subplot titles, and legends are applied. 
    top.set(xlabel = 'Usage = Rushing and Passing Attempts + Targets', ylabel = 'Total Season Fantasy Points')
    bottom.set(xlabel = 'Total Touchdowns', ylabel = 'Total Season Fantasy Points')
    top.set_title('Single Player fantasy points per season based upon usage.')
    bottom.set_title('Single Player fantasy points per season based upon Touchdowns.')
    top.legend(['QB','RB', 'WR', 'TE'])
    bottom.legend(['QB','RB', 'WR', 'TE'])

    # Adjusting margins of the subplots to ensure bottom title and top x-axis label dont overlap.
    metric_compare.subplots_adjust(top=0.88, bottom=0.11, left=0.11, right=0.9, hspace=0.5, wspace=0.2)

    # Saving png of the two subplots to the file directory.
    metric_compare.savefig('TDs_vs_Usage.png')


def age_pivot(PFR_data_all):
    '''
    The purpose of this function is to produce a pivot table based on the ages of players by position for fantasy value.

        Parameters:
            PFR_data_all (pandas.core.frame.DataFrame) : data array containing all merged and cleaned player data.

        Returns:
            n/a

    '''
    # Masking to prioritize displaying players with some fantasy football relevance as outside these bounds they are likely to be outliers that don't
    # represent fantasy relevant players. Games played column first needs to be converted from an index to a column.
    PFR_data_all['Fantasy', 'Games'] = PFR_data_all.index.to_frame()['Games']
    relevant_players = (PFR_data_all['Fantasy', 'FPts/G'] >= 1) & (PFR_data_all['Fantasy', 'Games'] > 2.5)
    PFR_data_fantasy = PFR_data_all[relevant_players]

    # Pivot table data is tabulated removing indexes and hierarchical ordering of columns for easier pivot table production. Only pertinent data to the pivot tables
    # is brought in after reducing index's.
    PFR_data_fantasy = PFR_data_fantasy.reset_index()
    data_pivot_table = pd.DataFrame({'Player': PFR_data_fantasy['Player',''], 'FPts/G': PFR_data_fantasy['Fantasy', 'FPts/G'], \
                                        'FPts': PFR_data_fantasy['Fantasy', 'FPts'], 'Pos': PFR_data_fantasy['Pos',''], 'Age': PFR_data_fantasy['Age','']})

    # Pivot table for fantasy points based on age and position is generated.
    age_pivot_fantasy = data_pivot_table.pivot_table('FPts/G', index='Pos', columns = 'Age', aggfunc='mean')

    # Pivot table for occurences based on combination of age and position is generated. This is to illustrate whether there is enough data to have
    # meaningful conclusions from the pivot table above.
    age_pivot_count = data_pivot_table.pivot_table(index='Pos', columns = 'Age', aggfunc='size')

    # Pivot tables are both masked to only include the fantasy relevant player positions.
    mask_qb_rb_wr_te = [len({age_pivot_fantasy.index.to_frame()['Pos'][i]} & {'QB', 'RB', 'WR', 'TE'}) > 0 for i in range(age_pivot_fantasy.shape[0])]
    age_pivot_fantasy = age_pivot_fantasy[mask_qb_rb_wr_te]
    age_pivot_count = age_pivot_count[mask_qb_rb_wr_te]

    # Header level 2
    print("Fantasy Points")
    # Printing fantasy point pivot tables to the console.
    print(age_pivot_fantasy)
    print()

    # Header level 2
    print("Occurences")
    # Printing occurence pivot tables to the console.
    print(age_pivot_count)


def draft_board(PFR_data_all, season_selection, team_amount_selection):
    '''
    The purpose of this function is to convert a large player dataset to a fantasy relevant dataset, including primarily only metrics
    that produce fantasy scoring, and associated results. To be tabulated for all positions and sorted by VOR.

        Parameters:
            PFR_data_pos (pandas.core.frame.DataFrame) : data array containing all merged and cleaned data, filtered to the user chosen position.

        Returns:
            PFR_data_fantasy_full (pandas.core.frame.DataFrame) : data array containing only fantasy point calculation relevant data, with value-over-replacement appended.

    '''

    # Create subsets for each fantasy position
    PFR_data_QB = PFR_data_all[(PFR_data_all.index.to_frame()['Pos'] == 'QB')]
    PFR_data_RB = PFR_data_all[(PFR_data_all.index.to_frame()['Pos'] == 'RB')]
    PFR_data_WB = PFR_data_all[(PFR_data_all.index.to_frame()['Pos'] == 'WR')]
    PFR_data_TE = PFR_data_all[(PFR_data_all.index.to_frame()['Pos'] == 'TE')]

    PFR_position_subsets = [PFR_data_QB, PFR_data_RB, PFR_data_WB, PFR_data_TE]

    # Data is then filtered to either year selection if the user does not choose 2019-2020
    idx = pd.IndexSlice
    for i, df in enumerate(PFR_position_subsets):
        if season_selection == '2020':
            PFR_position_subsets[i] = df.loc[idx[:,'2020', :, :, :, :], idx[:]]
        elif season_selection == '2019':
            PFR_position_subsets[i] = df.loc[idx[:,'2019', :, :, :, :], idx[:]]

        # The current positional data array is converted to a fantasy relevant data only.
        PFR_position_subsets[i] = convert_df_to_fantasy(PFR_position_subsets[i])

        if season_selection == "2019-2020":
            # If the user chooses to see both seasons, aggregation sum() on the fantasy relevant subset of the overall data, grouped by player.
            # Fantasy points per games must be dropped so they can be recalculated after summing the points.
            # PFR_position_subsets_midx = pd.MultiIndex.from_frame(PFR_position_subsets[i].reset_index().iloc[idx[:], idx[0:5]])

            PFR_position_subsets[i] = PFR_position_subsets[i].reset_index()
            player_position_dict = pd.Series(PFR_position_subsets[i].loc[idx[:], idx[('Pos', '')]].values, index = PFR_position_subsets[i].loc[idx[:], idx[('Player', '')]].values).to_dict()
            PFR_position_subsets[i] = PFR_position_subsets[i].drop([('Fantasy', 'FPts/G'), ('Season', ''), \
                                                                        ('Team', ''), ('Age', ''), ('Started', '')], axis=1)
            PFR_position_subsets[i] = PFR_position_subsets[i].groupby('Player').sum()
            PFR_position_subsets[i]['Pos'] = PFR_position_subsets[i].index.to_frame()['Player'].map(player_position_dict)
            PFR_position_subsets[i].set_index('Pos', append=True, inplace=True)

            # As noted fantasy points per game are recalculated, then we sort on this column.
            PFR_position_subsets[i]['Fantasy', 'FPts/G'] = PFR_position_subsets[i]['Fantasy', 'FPts'] / PFR_position_subsets[i]['Fantasy', 'Games']
            PFR_position_subsets[i] = PFR_position_subsets[i].sort_values(('Fantasy', 'FPts/G'), ascending=False)

        if PFR_position_subsets[i].index.to_frame()['Pos'][0] == 'RB' or PFR_position_subsets[i].index.to_frame()['Pos'][0] == 'WR':
            # If user selects to see RB or WR value over replacement is calculated using 3 replacements per team.
            vor_quantity = 3
            PFR_position_subsets[i]['Fantasy', 'VOR'] = PFR_position_subsets[i]['Fantasy', 'FPts/G'] - \
                                                            PFR_position_subsets[i].iloc[team_amount_selection*vor_quantity-1]['Fantasy', 'FPts/G']

        elif PFR_position_subsets[i].index.to_frame()['Pos'][0] == 'QB' or PFR_position_subsets[i].index.to_frame()['Pos'][0] == 'TE':
            # If user selects to see RB or WR value over replacement is calculated using 3 replacements per team.
            vor_quantity = 1
            PFR_position_subsets[i]['Fantasy', 'VOR'] = PFR_position_subsets[i]['Fantasy', 'FPts/G'] - \
                                                            PFR_position_subsets[i].iloc[team_amount_selection*vor_quantity-1]['Fantasy', 'FPts/G']
    
    draftboard = pd.concat(PFR_position_subsets)
    draftboard.set_index(('Fantasy', 'Games'), append=True, inplace=True)
    idx_before = draftboard.index.names
    idx_after = ["Games" if i == ('Fantasy', 'Games') else i for i in idx_before]
    draftboard.index.names = idx_after
    #df.reorder_levels(idx_after)
    draftboard = draftboard.sort_values(('Fantasy', 'VOR'), ascending=False)
    draftboard['Rank'] = [*range(1,draftboard.shape[0]+1)]
    print(draftboard)
    draftboard.to_excel("DraftBoard.xlsx")


def main():
    
    # File names within the data directory are placed within a list.
    all_file_names = [f for f in listdir('PFR Data Totals') if isfile(join('PFR Data Totals', f))]

    # A list to store the imported data is initialized.
    PFR_data = list()

    # All individual datasets are appended as objects into the PFR_data list.
    for i in all_file_names:
        PFR_data_add= import_data(i)
        PFR_data.append(PFR_data_add)

    # Data is merged into one large usable array.
    PFR_data_all = merge_and_clean_data(PFR_data)

    # Header level 0
    print()
    print("********** User Input **********")

    # Getting user input
    ppr_scoring, team_amount_selection, season_selection, index_selection, player_selection, position_selection = get_user_input(PFR_data_all)

    # Calculation of fantasy points based on scoring format and player stats.
    PFR_data_all = fantasy_calc(PFR_data_all, ppr_scoring)

    # Players are sorted by fantasy points per game to have the most relevant players presented first in the full dataset
    PFR_data_all = PFR_data_all.sort_values(('Fantasy', 'FPts/G'), ascending=False)
    
    # Creating masks for season, individual player, and position.
    mask_2020 = (PFR_data_all.index.to_frame()['Season'] == '2020')
    mask_2019 = (PFR_data_all.index.to_frame()['Season'] == '2019')
    mask_player = (PFR_data_all.index.to_frame()['Player'] == player_selection)
    mask_position = (PFR_data_all.index.to_frame()['Pos'] == position_selection)

    # Header level 0
    print()
    print("********** Player or Position Specific Stats **********")

    if index_selection == 'Player':

        # If user selects individual player stats, the main data array is filtered to that player.
        PFR_data_player = PFR_data_all[mask_player]

        if np.any([len({PFR_data_player.index.to_frame()['Pos'][i]} & {'QB'}) == 0 for i in range(PFR_data_player.shape[0])]):
            # If the selected player is not a quarterback, passing stats are dropped for better viewing experience.
            PFR_data_player = PFR_data_player.drop(['Passing'], axis=1) 
        elif np.any([len({PFR_data_player.index.to_frame()['Pos'][i]} & {'QB'}) > 0 for i in range(PFR_data_player.shape[0])]):
            # If the selected player is a quarterback, receiving stats are dropped for better viewing experience.
            PFR_data_player = PFR_data_player.drop(['Receiving'], axis=1)

        # Header level 1
        print(f"********** Position based stats for {player_selection} in the {season_selection} regular season(s) **********")
        print()

        # Printing player stats of either 2019, 2020, or both depending on which choice the user made.
        idx = pd.IndexSlice
        if season_selection == '2020':
            print(PFR_data_player.loc[idx[:,'2020', :, :, :, :], idx[:]])
        elif season_selection == '2019':
            print(PFR_data_player.loc[idx[:,'2019', :, :, :, :], idx[:]])
        else:
            print(PFR_data_player) 
    else:
        # If the user chose to view positional stats we mask to that positions, ie. RB, QB etc.
        PFR_data_pos = PFR_data_all[mask_position]

        # Data is then filtered to either year selection if the user does not choose 2019-2020
        idx = pd.IndexSlice
        if season_selection == '2020':
            PFR_data_pos = PFR_data_pos.loc[idx[:,'2020', :, :, :, :], idx[:]]
        elif season_selection == '2019':
            PFR_data_pos = PFR_data_pos.loc[idx[:,'2019', :, :, :, :], idx[:]]

        # Header level 1
        print(f"********** Summary of {position_selection} league-wide stats for the {season_selection} regular season(s) **********")
        print()

        # Data for the chosen position is summarized using describe() method.
        print(PFR_data_pos.describe())

        # The current positional data array is converted to a fantasy relevant data only.
        PFR_data_fan = convert_df_to_fantasy(PFR_data_pos)
        if season_selection == "2019-2020":
            # If the user chooses to see both seasons, aggregation sum() on the fantasy relevant subset of the overall data, grouped by player.
            # Fantasy points per games must be dropped so they can be recalculated after summing the points.
            PFR_data_fan = PFR_data_fan.drop([('Fantasy', 'FPts/G')], axis=1)
            PFR_data_fan = PFR_data_fan.groupby('Player').sum()

            # As noted fantasy points per game are recalculated, then we sort on this column.
            PFR_data_fan['Fantasy', 'FPts/G'] = PFR_data_fan['Fantasy', 'FPts'] / PFR_data_fan['Fantasy', 'Games']
            PFR_data_fan = PFR_data_fan.sort_values(('Fantasy', 'FPts/G'), ascending=False)


        if position_selection == 'RB' or position_selection == 'WR':
            # If user selects to see RB or WR value over replacement is calculated using 3 replacements per team.
            vor_quantity = 3
            PFR_data_fan['Fantasy', 'VOR'] = PFR_data_fan['Fantasy', 'FPts/G'] - PFR_data_fan.iloc[team_amount_selection*vor_quantity-1]['Fantasy', 'FPts/G']
        elif position_selection == 'QB' or position_selection == 'TE':
            # If user selects to see RB or WR value over replacement is calculated using 3 replacements per team.
            vor_quantity = 1
            PFR_data_fan['Fantasy', 'VOR'] = PFR_data_fan['Fantasy', 'FPts/G'] - PFR_data_fan.iloc[team_amount_selection*vor_quantity-1]['Fantasy', 'FPts/G']
        
        # Header level 1
        print()
        print(f"********** {position_selection} fantasy stats for the {season_selection} regular season(s) (min 4 FPts/G and 3 games played) **********")
        print()

        # Position specific stats for the chosen season are printed.
        print(PFR_data_fan)

    # Header level 0
    print()
    print("********** All Data Stats **********")

    # Header level 1
    print("********** Summary of all stats for the 2019 and 2020 regular season(s) **********")
    print()

    # Describe is used to present a summary of the full merged data set.
    print(PFR_data_all.describe())

    # Header level 1
    print()
    print(f"********** Fantasy Scores based upon age and position **********")
    print()

    # Age pivot tables are printed.
    age_pivot(PFR_data_all)

    # Fantasy point trends are produced in the popup window and saved to the file directory as "TDs_vs_Usage.png"
    position_trends(PFR_data_all)

    # Header level 0
    print()
    print("Full merged stat data can be found in the file directory under AllData.xlsx")
    print("Plots have been generated from the full dataset in the popup window \"Figure 1\" demonstrating the importance of both usage and the ability to score touchdowns.")
    print()

    # Exporting to excel for optional viewing and then plots are shown.
    PFR_data_all.to_excel('AllData.xlsx')
    #plt.show()
    
    ############# AFTER FINAL PROJECT COMPLETED ####################

    draft_board(PFR_data_all, season_selection, team_amount_selection)


if __name__ == '__main__':
    main()