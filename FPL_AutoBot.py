import requests
# from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import pulp
import sys
import os
from understatapi import UnderstatClient
# from fuzzywuzzy import fuzz

# ==============================================================================
def select_team(full_df, sub_factor=0.2, total_budget=100):
	num_players = len(full_df)

	model = pulp.LpProblem('Team_Selection', pulp.LpMaximize)

	decisions = [
		pulp.LpVariable(f"x{i}", lowBound=0, upBound=1, cat='Integer')
		for i in range(num_players)
	]
	captain_decisions = [
		pulp.LpVariable(f"y{i}", lowBound=0, upBound=1, cat='Integer')
		for i in range(num_players)
	]
	sub_decisions = [
		pulp.LpVariable(f"z{i}", lowBound=0, upBound=1, cat='Integer')
		for i in range(num_players)
	]

	# objective function:
	# model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * full_df.iloc[i]['roi'] for i in range(num_players)), "Objective"
	# model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * full_df.iloc[i]['ROI/EG + ROI/ECS'] for i in range(num_players)), "Objective"
	# model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * full_df.iloc[i]['Form_adjusted'] for i in range(num_players)), "Objective"
	# model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * full_df.iloc[i]['form'] for i in range(num_players)), "Objective"
	# model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * full_df.iloc[i]['Form_adjusted_ROI']for i in range(num_players)), "Objective"
	# model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * full_df.iloc[i]['now_cost']for i in range(num_players)), "Objective"
	# model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * full_df.iloc[i]['Form_adjusted_cost']for i in range(num_players)), "Objective"
	# model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * full_df.iloc[i]['EG_6wk']for i in range(num_players)), "Objective"
	model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * full_df.iloc[i]['Final_value']for i in range(num_players)), "Objective"
	# model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * full_df.iloc[i]['total_points']for i in range(num_players)), "Objective"

	# constraints
	# total budget
	model += sum((decisions[i] + sub_decisions[i]) * full_df.iloc[i]['now_cost'] for i in range(num_players)) <= total_budget  # total cost

	# position constraints
	# 1 goalkeeper
	model += sum(decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'GKP') == 1
	# 2 total goalkeepers
	model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'GKP') == 2

	# 3-5 starting defenders
	model += sum(decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'DEF') >= 3
	model += sum(decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'DEF') <= 5
	# 5 total defenders
	model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'DEF') == 5

	# 3-5 starting midfielders
	model += sum(decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'MID') >= 3
	model += sum(decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'MID') <= 5
	# 5 total midfielders
	model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'MID') == 5

	# 2-3 starting attackers
	model += sum(decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'FWD') >= 2
	model += sum(decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'FWD') <= 3
	# 3 total attackers
	model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'FWD') == 3

	# club constraint
	for club_id in full_df.team.unique():
		model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if full_df['team'].iloc[i] == club_id) <= 3  # max 3 players

	model += sum(decisions) == 11  # total team size
	model += sum(captain_decisions) == 1  # 1 captain
	model += sum(sub_decisions) == 4
	# model += sum(captain_decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'FWD') >= 1 #captain must be a forward
	model += sum(captain_decisions[i] for i in range(num_players) if full_df['position'].iloc[i] == 'MID') >= 1 #captain must be a midfielder

	for i in range(num_players):
		model += (decisions[i] - captain_decisions[i]) >= 0  # captain must also be on team
		model += (decisions[i] + sub_decisions[i]) <= 1  # subs must not be on team

	model.solve()
	return decisions, captain_decisions, sub_decisions
# ==============================================================================
def get_data_fpl(url="https://fantasy.premierleague.com/api/bootstrap-static/"):
	print("Getting official FPL stats ...")
	r = requests.get(url)
	json = r.json()
	return pd.DataFrame(json['elements'])
# ==============================================================================
def get_data_understat(season="2024"):
	print("Getting Understat data ...")
	understat = UnderstatClient()
	return understat.league(league="EPL").get_player_data(season)
# ==============================================================================
def create_adjusted_feature(df, feature_to_adjust, reversed=False):
	if reversed:
		df[f"GKP_{feature_to_adjust}_adjusted"] = 1.0 - df[df["position"] == "GKP"][f"{feature_to_adjust}"]/df[df["position"] == "GKP"][f"{feature_to_adjust}"].max()
		df[f"GKP_{feature_to_adjust}_adjusted"] = df[f"GKP_{feature_to_adjust}_adjusted"].fillna(0)
		df[f"DEF_{feature_to_adjust}_adjusted"] = 1.0 - df[df["position"] == "DEF"][f"{feature_to_adjust}"]/df[df["position"] == "DEF"][f"{feature_to_adjust}"].max()
		df[f"DEF_{feature_to_adjust}_adjusted"] = df[f"DEF_{feature_to_adjust}_adjusted"].fillna(0)
		df[f"MID_{feature_to_adjust}_adjusted"] = 1.0 - df[df["position"] == "MID"][f"{feature_to_adjust}"]/df[df["position"] == "MID"][f"{feature_to_adjust}"].max()
		df[f"MID_{feature_to_adjust}_adjusted"] = df[f"MID_{feature_to_adjust}_adjusted"].fillna(0)
		df[f"FWD_{feature_to_adjust}_adjusted"] = 1.0 - df[df["position"] == "FWD"][f"{feature_to_adjust}"]/df[df["position"] == "FWD"][f"{feature_to_adjust}"].max()
	else:
		df[f"GKP_{feature_to_adjust}_adjusted"] = df[df["position"] == "GKP"][f"{feature_to_adjust}"]/df[df["position"] == "GKP"][f"{feature_to_adjust}"].max()
		df[f"GKP_{feature_to_adjust}_adjusted"] = df[f"GKP_{feature_to_adjust}_adjusted"].fillna(0)
		df[f"DEF_{feature_to_adjust}_adjusted"] = df[df["position"] == "DEF"][f"{feature_to_adjust}"]/df[df["position"] == "DEF"][f"{feature_to_adjust}"].max()
		df[f"DEF_{feature_to_adjust}_adjusted"] = df[f"DEF_{feature_to_adjust}_adjusted"].fillna(0)
		df[f"MID_{feature_to_adjust}_adjusted"] = df[df["position"] == "MID"][f"{feature_to_adjust}"]/df[df["position"] == "MID"][f"{feature_to_adjust}"].max()
		df[f"MID_{feature_to_adjust}_adjusted"] = df[f"MID_{feature_to_adjust}_adjusted"].fillna(0)
		df[f"FWD_{feature_to_adjust}_adjusted"] = df[df["position"] == "FWD"][f"{feature_to_adjust}"]/df[df["position"] == "FWD"][f"{feature_to_adjust}"].max()
	df[f"FWD_{feature_to_adjust}_adjusted"] = df[f"FWD_{feature_to_adjust}_adjusted"].fillna(0)
	df[f"{feature_to_adjust}_adj"] = df[f"GKP_{feature_to_adjust}_adjusted"]+df[f"DEF_{feature_to_adjust}_adjusted"]+df[f"MID_{feature_to_adjust}_adjusted"]+df[f"FWD_{feature_to_adjust}_adjusted"]
	df.drop(f'GKP_{feature_to_adjust}_adjusted', axis=1, inplace=True)
	df.drop(f'DEF_{feature_to_adjust}_adjusted', axis=1, inplace=True)
	df.drop(f'MID_{feature_to_adjust}_adjusted', axis=1, inplace=True)
	df.drop(f'FWD_{feature_to_adjust}_adjusted', axis=1, inplace=True)

	return(df)
# ==============================================================================
# Get the FPL data table
df = get_data_fpl(url="https://fantasy.premierleague.com/api/bootstrap-static/")
# df = pd.read_csv("df_fpl.csv")

# Get my current team
# dm = pd.read_csv("df_myteam.csv")

# Get the Understat data table
# du = pd.DataFrame(get_data_understat(season="2023"))

# Replace the numerical value for the clubs for the real names of the clubs
df.team = df.team.replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
			["Arsenal", "Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea",
			"Palace", "Everton", "Fullham", "Ipswich", "Leicester", "Liverpool", 
			"Man City", "Man United", "Newcastle", "Forest", "Southampton",
			"Spurs", "West Ham", "Wolves"])
# Create a new column of position instead of the unwieldy "element_type" column
df['position'] = df['element_type'].replace([1,2,3,4], ["GKP", "DEF", "MID", "FWD"])

print("Total number of players:", len(df))

# sys.exit()

# Make Haaland and Salah unavailable
# df.loc[df["web_name"] == "Haaland", "status"] = "u"
# df.loc[df["web_name"] == "M.Salah", "status"] = "u"
# df.drop(df[df['web_name'] == "M.Salah"].index, inplace=True)
# df.drop(df[df['web_name'] == "Haaland"].index, inplace=True)

# Select only players that are available
df = df[df['status'] == 'a']
print("Total number of available players:", len(df))

# Select only players that have more than am certain number of minutes played this season so far
# num_min_played = 0
num_min_played = 90*(30/2)
df = df[df['minutes'] >= num_min_played]
print(
	f"Total number of FPL players with more than {str(num_min_played)} minutes of playtime:",
	len(df),
)

# Match each player in FPL table with players in Understat table using some fuzzy logic
# df["full_name"] = df["first_name"]+' '+df["second_name"]
# df['name_team_df'] = df['full_name']+' '+df['team']

# Select players from certain team only (for possible blank gameweeks during the season)
# df.drop(df[df['team'] == 'Arsenal'].index, inplace=True)
# df.drop(df[df['team'] == 'Villa'].index, inplace=True)
# df.drop(df[df['team'] == 'Bournemouth'].index, inplace=True)
# df.drop(df[df['team'] == 'Brentford'].index, inplace=True)
# df.drop(df[df['team'] == 'Brighton'].index, inplace=True)
# df.drop(df[df['team'] == 'Chelsea'].index, inplace=True)
# df.drop(df[df['team'] == 'Palace'].index, inplace=True)
# df.drop(df[df['team'] == 'Everton'].index, inplace=True)
# df.drop(df[df['team'] == 'Fullham'].index, inplace=True)
# df.drop(df[df['team'] == 'Ipswich'].index, inplace=True)
# df.drop(df[df['team'] == 'Leicester'].index, inplace=True)
# df.drop(df[df['team'] == 'Liverpool'].index, inplace=True)
# df.drop(df[df['team'] == 'Man City'].index, inplace=True)
# df.drop(df[df['team'] == 'Man United'].index, inplace=True)
# df.drop(df[df['team'] == 'Newcastle'].index, inplace=True)
# df.drop(df[df['team'] == 'Forest'].index, inplace=True)
# df.drop(df[df['team'] == 'Southampton'].index, inplace=True)
# df.drop(df[df['team'] == 'Spurs'].index, inplace=True)
# df.drop(df[df['team'] == 'West Ham'].index, inplace=True)
# df.drop(df[df['team'] == 'Wolves'].index, inplace=True)

# sys.exit()

full_df = df.loc[:, ["web_name", "position", "team", "now_cost", "minutes",
                    "total_points", "bonus", "bps", "selected_by_percent",
					"goals_scored", "assists",
					"expected_goals", "expected_assists", "expected_goal_involvements",
					"expected_goals_per_90", "expected_assists_per_90", "expected_goal_involvements_per_90",
					"influence", "creativity", "threat", "form",
					"ep_this", "ep_next", "transfers_in_event", "transfers_out_event"]]

# sys.exit()

# Some columns are strings that should be numbers. Do the conversion here
full_df['now_cost'] = pd.to_numeric(full_df['now_cost'])
full_df['minutes'] = pd.to_numeric(full_df['minutes'])
full_df['total_points'] = pd.to_numeric(full_df['total_points'])
full_df['total_points'] = pd.to_numeric(full_df['total_points'])
full_df['bps'] = pd.to_numeric(full_df['bps'])
full_df['selected_by_percent'] = pd.to_numeric(full_df['selected_by_percent'])
full_df['goals_scored'] = pd.to_numeric(full_df['goals_scored'])
full_df['assists'] = pd.to_numeric(full_df['assists'])
full_df['expected_goals'] = pd.to_numeric(full_df['expected_goals'])
full_df['expected_assists'] = pd.to_numeric(full_df['expected_assists'])
full_df['expected_goal_involvements'] = pd.to_numeric(full_df['expected_goal_involvements'])
full_df['influence'] = pd.to_numeric(full_df['influence'])
full_df['creativity'] = pd.to_numeric(full_df['creativity'])
full_df['threat'] = pd.to_numeric(full_df['threat'])
full_df['form'] = pd.to_numeric(full_df['form'])
full_df['ep_this'] = pd.to_numeric(full_df['ep_this'])
full_df['ep_next'] = pd.to_numeric(full_df['ep_next'])
full_df['transfers_in_event'] = pd.to_numeric(full_df['transfers_in_event'])
full_df['transfers_out_event'] = pd.to_numeric(full_df['transfers_out_event'])
# full_df[''] = pd.to_numeric(full_df[''])
# full_df[''] = pd.to_numeric(full_df[''])
# full_df[''] = pd.to_numeric(full_df[''])
# full_df[''] = pd.to_numeric(full_df[''])
# full_df[''] = pd.to_numeric(full_df[''])
# full_df[''] = pd.to_numeric(full_df[''])
# full_df[''] = pd.to_numeric(full_df[''])
# full_df[''] = pd.to_numeric(full_df[''])
# full_df[''] = pd.to_numeric(full_df[''])
# full_df[''] = pd.to_numeric(full_df[''])
# full_df[''] = pd.to_numeric(full_df[''])

# Scale the now_cost columns so that it lines up with the FPL website numbers
full_df['now_cost'] = full_df['now_cost']/10.0
full_df['transfers_net'] = full_df['transfers_in_event']-full_df['transfers_out_event']

# Create some new columns that might help with team selection
full_df['points_per_cost'] = full_df['total_points']/full_df['now_cost']
full_df['points_per_minute'] = full_df['total_points']/full_df['minutes']
full_df['bonus_per_cost'] = full_df['bonus']/full_df['now_cost']
full_df['bonus_per_minute'] = full_df['bonus']/full_df['minutes']
full_df['bps_per_cost'] = full_df['bps']/full_df['now_cost']
full_df['bps_per_minute'] = full_df['bps']/full_df['minutes']
full_df['expected_goals_per_cost'] = full_df['expected_goals']/full_df['now_cost']
full_df['expected_goals_per_minute'] = full_df['expected_goals']/full_df['minutes']
full_df['expected_assists_per_cost'] = full_df['expected_assists']/full_df['now_cost']
full_df['expected_assists_per_minute'] = full_df['expected_assists']/full_df['minutes']
full_df['expected_goal_involvements_per_cost'] = full_df['expected_goal_involvements']/full_df['now_cost']
full_df['expected_goal_involvements_per_minute'] = full_df['expected_goal_involvements']/full_df['minutes']
full_df['influence_per_cost'] = full_df['influence']/full_df['now_cost']
full_df['influence_per_minute'] = full_df['influence']/full_df['minutes']
full_df['creativity_per_cost'] = full_df['creativity']/full_df['now_cost']
full_df['creativity_per_minute'] = full_df['creativity']/full_df['minutes']
full_df['threat_per_cost'] = full_df['threat']/full_df['now_cost']
full_df['threat_per_minute'] = full_df['threat']/full_df['minutes']
full_df['form_per_cost'] = full_df['form']/full_df['now_cost']
full_df['form_per_minute'] = full_df['form']/full_df['minutes']
full_df['ep_this_per_cost'] = full_df['ep_this']/full_df['now_cost']
full_df['ep_this_per_minute'] = full_df['ep_this']/full_df['minutes']
full_df['ep_next_per_cost'] = full_df['ep_next']/full_df['now_cost']
full_df['ep_next_per_minute'] = full_df['ep_next']/full_df['minutes']
full_df['goals_scored_per_cost'] = full_df['goals_scored']/full_df['now_cost']
full_df['goals_scored_per_minute'] = full_df['goals_scored']/full_df['minutes']
full_df['assists_per_cost'] = full_df['assists']/full_df['now_cost']
full_df['assists_per_minute'] = full_df['assists']/full_df['minutes']
full_df["goals_over_expected"] = np.abs(full_df["goals_scored"] - full_df["expected_goals"])
full_df["assists_over_expected"] = np.abs(full_df["assists"] - full_df["expected_assists"])
# full_df['_per_cost'] = full_df['']/full_df['now_cost']
# full_df['_per_minute'] = full_df['']/full_df['minutes']

# Create all the adjusted features
full_df = create_adjusted_feature(full_df, "minutes", reversed=False)
# full_df = create_adjusted_feature(full_df, "selected_by_percent", reversed=False)
full_df = create_adjusted_feature(full_df, "total_points", reversed=False)
full_df = create_adjusted_feature(full_df, "points_per_cost", reversed=False)
full_df = create_adjusted_feature(full_df, "points_per_minute", reversed=False)
full_df = create_adjusted_feature(full_df, "bonus", reversed=False)
full_df = create_adjusted_feature(full_df, "bonus_per_cost", reversed=False)
full_df = create_adjusted_feature(full_df, "bonus_per_minute", reversed=False)
full_df = create_adjusted_feature(full_df, "bps", reversed=False)
full_df = create_adjusted_feature(full_df, "bps_per_cost", reversed=False)
full_df = create_adjusted_feature(full_df, "bps_per_minute", reversed=False)
full_df = create_adjusted_feature(full_df, "expected_goals", reversed=False)
full_df = create_adjusted_feature(full_df, "expected_goals_per_cost", reversed=False)
full_df = create_adjusted_feature(full_df, "expected_goals_per_minute", reversed=False)
full_df = create_adjusted_feature(full_df, "expected_assists", reversed=False)
full_df = create_adjusted_feature(full_df, "expected_assists_per_cost", reversed=False)
full_df = create_adjusted_feature(full_df, "expected_assists_per_minute", reversed=False)
full_df = create_adjusted_feature(full_df, "expected_goal_involvements", reversed=False)
full_df = create_adjusted_feature(full_df, "expected_goal_involvements_per_cost", reversed=False)
full_df = create_adjusted_feature(full_df, "expected_goal_involvements_per_minute", reversed=False)
# full_df = create_adjusted_feature(full_df, "influence", reversed=False)
# full_df = create_adjusted_feature(full_df, "influence_per_cost", reversed=False)
# full_df = create_adjusted_feature(full_df, "influence_per_minute", reversed=False)
# full_df = create_adjusted_feature(full_df, "creativity", reversed=False)
# full_df = create_adjusted_feature(full_df, "creativity_per_cost", reversed=False)
# full_df = create_adjusted_feature(full_df, "creativity_per_minute", reversed=False)
# full_df = create_adjusted_feature(full_df, "threat", reversed=False)
# full_df = create_adjusted_feature(full_df, "threat_per_cost", reversed=False)
# full_df = create_adjusted_feature(full_df, "threat_per_minute", reversed=False)
# full_df = create_adjusted_feature(full_df, "form", reversed=False)
# full_df = create_adjusted_feature(full_df, "form_per_cost", reversed=False)
# full_df = create_adjusted_feature(full_df, "form_per_minute", reversed=False)
# full_df = create_adjusted_feature(full_df, "ep_this", reversed=False)
# full_df = create_adjusted_feature(full_df, "ep_this_per_cost", reversed=False)
# full_df = create_adjusted_feature(full_df, "ep_this_per_minute", reversed=False)
# full_df = create_adjusted_feature(full_df, "ep_next", reversed=False)
# full_df = create_adjusted_feature(full_df, "ep_next_per_cost", reversed=False)
# full_df = create_adjusted_feature(full_df, "ep_next_per_minute", reversed=False)
full_df = create_adjusted_feature(full_df, "goals_scored", reversed=False)
full_df = create_adjusted_feature(full_df, "goals_scored_per_cost", reversed=False)
full_df = create_adjusted_feature(full_df, "goals_scored_per_minute", reversed=False)
full_df = create_adjusted_feature(full_df, "assists", reversed=False)
full_df = create_adjusted_feature(full_df, "assists_per_cost", reversed=False)
full_df = create_adjusted_feature(full_df, "assists_per_minute", reversed=False)
full_df = create_adjusted_feature(full_df, "goals_over_expected", reversed=True)
full_df = create_adjusted_feature(full_df, "assists_over_expected", reversed=True)
# full_df = create_adjusted_feature(full_df, "transfers_net", reversed=False)
# full_df = create_adjusted_feature(full_df, "transfers_in_event", reversed=False)
# full_df = create_adjusted_feature(full_df, "transfers_out_event", reversed=True)
# full_df = create_adjusted_feature(full_df, "", reversed=False)
# full_df = create_adjusted_feature(full_df, "", reversed=False)


# Drop some columns that we don't need anymore
# full_df.drop('web_name', axis=1, inplace=True)
# full_df.drop('position', axis=1, inplace=True)
# full_df.drop('team', axis=1, inplace=True)
# full_df.drop('minutes', axis=1, inplace=True)
# full_df.drop('total_points', axis=1, inplace=True)
# full_df.drop('selected_by_percent', axis=1, inplace=True)
full_df.drop('points_per_cost', axis=1, inplace=True)
full_df.drop('points_per_minute', axis=1, inplace=True)
full_df.drop('bonus', axis=1, inplace=True)
full_df.drop('bonus_per_cost', axis=1, inplace=True)
full_df.drop('bonus_per_minute', axis=1, inplace=True)
full_df.drop('bps', axis=1, inplace=True)
full_df.drop('bps_per_cost', axis=1, inplace=True)
full_df.drop('bps_per_minute', axis=1, inplace=True)
full_df.drop('expected_goals', axis=1, inplace=True)
full_df.drop('expected_goals_per_cost', axis=1, inplace=True)
full_df.drop('expected_goals_per_minute', axis=1, inplace=True)
full_df.drop('expected_goals_per_90', axis=1, inplace=True)
full_df.drop('expected_assists', axis=1, inplace=True)
full_df.drop('expected_assists_per_cost', axis=1, inplace=True)
full_df.drop('expected_assists_per_minute', axis=1, inplace=True)
full_df.drop('expected_assists_per_90', axis=1, inplace=True)
full_df.drop('expected_goal_involvements', axis=1, inplace=True)
full_df.drop('expected_goal_involvements_per_cost', axis=1, inplace=True)
full_df.drop('expected_goal_involvements_per_minute', axis=1, inplace=True)
full_df.drop('expected_goal_involvements_per_90', axis=1, inplace=True)
full_df.drop('influence', axis=1, inplace=True)
full_df.drop('influence_per_cost', axis=1, inplace=True)
full_df.drop('influence_per_minute', axis=1, inplace=True)
full_df.drop('creativity', axis=1, inplace=True)
full_df.drop('creativity_per_cost', axis=1, inplace=True)
full_df.drop('creativity_per_minute', axis=1, inplace=True)
full_df.drop('threat', axis=1, inplace=True)
full_df.drop('threat_per_cost', axis=1, inplace=True)
full_df.drop('threat_per_minute', axis=1, inplace=True)
full_df.drop('form', axis=1, inplace=True)
full_df.drop('form_per_cost', axis=1, inplace=True)
full_df.drop('form_per_minute', axis=1, inplace=True)
full_df.drop('ep_this', axis=1, inplace=True)
full_df.drop('ep_this_per_cost', axis=1, inplace=True)
full_df.drop('ep_this_per_minute', axis=1, inplace=True)
full_df.drop('ep_next', axis=1, inplace=True)
full_df.drop('ep_next_per_cost', axis=1, inplace=True)
full_df.drop('ep_next_per_minute', axis=1, inplace=True)
full_df.drop('goals_scored', axis=1, inplace=True)
full_df.drop('goals_scored_per_cost', axis=1, inplace=True)
full_df.drop('goals_scored_per_minute', axis=1, inplace=True)
full_df.drop('assists', axis=1, inplace=True)
full_df.drop('assists_per_cost', axis=1, inplace=True)
full_df.drop('assists_per_minute', axis=1, inplace=True)
full_df.drop('transfers_in_event', axis=1, inplace=True)
full_df.drop('transfers_out_event', axis=1, inplace=True)
full_df.drop('transfers_net', axis=1, inplace=True)
full_df.drop('goals_over_expected', axis=1, inplace=True)
full_df.drop('assists_over_expected', axis=1, inplace=True)
# full_df.drop('', axis=1, inplace=True)
# full_df.drop('', axis=1, inplace=True)

# Add all the adjusted / scaled features together for a much better reflection of a players ability in a specific position
matches = [match for match in full_df.columns if "_adj" in match]
# full_df['adj_fts'] = full_df.loc[:, matches].sum(axis=1)

# ******************************************************************************
# Creating the Final value for each player can be done in a few different ways
# This "Final Value" will be used to select the best team using the backpack optimization problem
# ******************************************************************************
# This final value does NOT take the other FPL managers into account - Could we find value where others don't?
full_df['Final_value'] = full_df.loc[:, matches].sum(axis=1)
# full_df['Final_value'] = full_df['adj_fts'] #This one chooses a much better team for the long run using all the available stats
# full_df.drop('adj_fts', axis=1, inplace=True)

# pd.set_option("max_colwidth", 2)
pd.options.display.float_format = "{:,.2f}".format
# ******************************************************************************
# Print out a list of the players for each position sorted by the "Final value" we created
# ******************************************************************************
GKP = full_df[full_df["position"] == "GKP"]
GKP = GKP.sort_values(by = 'Final_value', ascending = False)
print("************************************************************************************ GOALKEEPERS ************************************************************************************")
print(GKP.head(14))
GKP.to_csv("df_GKP.csv", index=False)

DEF = full_df[full_df["position"] == "DEF"]
DEF = DEF.sort_values(by = 'Final_value', ascending = False)
print("************************************************************************************ DEFENDERS ************************************************************************************")
print(DEF.head(24))
DEF.to_csv("df_DEF.csv", index=False)

MID = full_df[full_df["position"] == "MID"]
MID = MID.sort_values(by = 'Final_value', ascending = False)
print("************************************************************************************ MIDFIELDERS ************************************************************************************")
print(MID.head(24))
MID.to_csv("df_MID.csv", index=False)

FWD = full_df[full_df["position"] == "FWD"]
FWD = FWD.sort_values(by = 'Final_value', ascending = False)
print("************************************************************************************ FORWARDS ************************************************************************************")
print(FWD.head(24))
FWD.to_csv("df_FWD.csv", index=False)

print("**********************************************************************************************************************************************************************************")
# dfs = [GKP[:17], DEF[:17], MID[:17], FWD[:17]]
# final_df = pd.concat(dfs)
# print(final_df.to_markdown())
# sys.exit()


# ******************************************************************************
# Let an algorithm decide which is the best team to select by solving the knapsack problem.
# The sub_factor controls how much of an influence we want our subs to have on the final team selection.
# Making this very low will select cheap players and focus on more expensive palyers for the first team.
# ******************************************************************************
subfac = 1.0
decisions, captain_decisions, sub_decisions = select_team(full_df, sub_factor=subfac, total_budget=102.8)

print()
print("****************************************************************************************** FIRST TEAM ******************************************************************************************")
player_indices = [
	i for i in range(len(decisions)) if decisions[i].value() == 1
]
# print("Best First Team Selection:")
FT = full_df.iloc[player_indices]
print(FT.sort_values(by="position", ascending=False))

print()
print("****************************************************************************************** SUBS ******************************************************************************************")
sub_player_indices = [
	i for i in range(len(sub_decisions)) if sub_decisions[i].value() == 1
]
# print("Subs:")
SB = full_df.iloc[sub_player_indices]
print(SB.sort_values(by="position", ascending=False))

print()
print("****************************************************************************************** CAPTAIN ******************************************************************************************")
cap_player_indices = [
	i
	for i in range(len(captain_decisions))
	if captain_decisions[i].value() == 1
]
# print("Captain:")
CP = full_df.iloc[cap_player_indices]
print(CP.sort_values(by="position", ascending=False))

print()
print("*************************************************************************")
print()
print('Total team cost:', np.round(np.sum(FT["now_cost"])+np.sum(SB["now_cost"]), decimals=2))
print('Total team points:', np.round(np.sum(FT["total_points"])+np.sum(CP["total_points"]), decimals=2))
# print('Total ExpectPts (ex. bench):',np.round(np.sum(FT["ep_next"]), decimals=2))
# print('Total ExpectPts (inc bench):',np.round(np.sum(FT["ep_next"])+np.sum(SB["ep_next"]), decimals=2))

sys.exit()