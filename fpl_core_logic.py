import requests
# from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import pulp
# from fuzzywuzzy import fuzz

# ==============================================================================
def select_team(full_df, sub_factor=0.2, total_budget=100):
	"""
	Selects the optimal FPL team based on the provided player data, budget, and constraints.
	Returns the DataFrames for the first team, substitutes, and captain.
	"""
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

	# objective function: Maximize Final_value
	model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i] * sub_factor) * full_df.iloc[i]['Final_value'] for i in range(num_players)), "Objective"

	# constraints
	# total budget
	model += sum((decisions[i] + sub_decisions[i]) * full_df.iloc[i]['now_cost'] for i in range(num_players)) <= total_budget

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

	# Extract selected player dataframes
	player_indices = [i for i in range(num_players) if decisions[i].value() == 1]
	sub_player_indices = [i for i in range(num_players) if sub_decisions[i].value() == 1]
	cap_player_indices = [i for i in range(num_players) if captain_decisions[i].value() == 1]

	first_team_df = full_df.iloc[player_indices].sort_values(by="position", ascending=False)
	subs_df = full_df.iloc[sub_player_indices].sort_values(by="position", ascending=False)
	captain_df = full_df.iloc[cap_player_indices].sort_values(by="position", ascending=False)

	return first_team_df, subs_df, captain_df
# ==============================================================================
def get_data_fpl(url="https://fantasy.premierleague.com/api/bootstrap-static/"):
	print("Getting official FPL stats ...")
	try:
		r = requests.get(url, timeout=10)
		r.raise_for_status()
		json = r.json()
		return pd.DataFrame(json['elements'])
	except requests.exceptions.RequestException as e:
		print(f"Error fetching FPL data: {e}")
		return None
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
def process_data(df):
	"""Processes the raw FPL data to prepare it for team selection."""
	if df is None or df.empty:
		print("Input DataFrame is empty or None. Skipping processing.")
		return None

	print("Processing FPL data...")

	# Replace team IDs with names
	team_map = {
		1: "Arsenal", 2: "Villa", 3: "Bournemouth", 4: "Brentford", 5: "Brighton",
		6: "Chelsea", 7: "Palace", 8: "Everton", 9: "Fullham", 10: "Ipswich",
		11: "Leicester", 12: "Liverpool", 13: "Man City", 14: "Man United",
		15: "Newcastle", 16: "Forest", 17: "Southampton", 18: "Spurs",
		19: "West Ham", 20: "Wolves"
	}
	# Handle potential missing team IDs gracefully
	df['team'] = df['team'].map(team_map).fillna('Unknown Team')


	# Replace element type with position name
	position_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
	df['position'] = df['element_type'].map(position_map).fillna('Unknown Pos')

	# Filter unavailable players (optional, can be done later based on user choice)
	df = df[df['status'] == 'a'].copy() # Use .copy() to avoid SettingWithCopyWarning
	print(f"Available players: {len(df)}")

	# Filter by minutes played (optional, could be a user setting)
	# num_min_played = 90 * (30 / 2) # Example: half season equivalent
	num_min_played = 0 # Setting to 0 includes all available players for now
	df = df[df['minutes'] >= num_min_played].copy()
	print(f"Players with >= {num_min_played} minutes: {len(df)}")

	# Select and convert necessary columns
	required_cols = ["web_name", "position", "team", "now_cost", "minutes",
					 "total_points", "bonus", "bps", "selected_by_percent",
					 "goals_scored", "assists", "expected_goals", "expected_assists",
					 "expected_goal_involvements", "influence", "creativity",
					 "threat", "form", "ep_this", "ep_next", "transfers_in_event",
					 "transfers_out_event"]

	# Check if all required columns exist
	missing_cols = [col for col in required_cols if col not in df.columns]
	if missing_cols:
		print(f"Warning: Missing required columns in FPL data: {missing_cols}")
		# Decide how to handle: return None, raise error, or proceed with available columns
		# For now, let's filter out missing columns and proceed
		required_cols = [col for col in required_cols if col in df.columns]
		if not required_cols:
			 print("Error: No required columns found.")
			 return None

	full_df = df.loc[:, required_cols].copy()

	numeric_cols = [col for col in required_cols if col not in ["web_name", "position", "team"]]
	for col in numeric_cols:
		full_df[col] = pd.to_numeric(full_df[col], errors='coerce') # errors='coerce' turns unparseable values into NaN

	# Handle potential NaNs created by 'coerce' or already present
	# Option 1: Fill with 0 (simple approach)
	full_df.fillna(0, inplace=True)
	# Option 2: More sophisticated imputation (mean, median) if appropriate

	# Scale cost
	if 'now_cost' in full_df.columns:
		full_df['now_cost'] = full_df['now_cost'] / 10.0

	# Feature Engineering - Safely calculate new features
	# Check for division by zero or NaN in denominators
	def safe_divide(numerator_col, denominator_col, df_in):
		num = df_in[numerator_col]
		den = df_in[denominator_col]
		# Create a safe division result, returning 0 where denominator is 0 or NaN
		return (num / den.replace(0, np.nan)).fillna(0)

	if 'total_points' in full_df.columns and 'now_cost' in full_df.columns:
		full_df['points_per_cost'] = safe_divide('total_points', 'now_cost', full_df)
	if 'total_points' in full_df.columns and 'minutes' in full_df.columns:
		full_df['points_per_minute'] = safe_divide('total_points', 'minutes', full_df)
	# ... add similar safe calculations for all _per_cost and _per_minute features ...
	if 'bonus' in full_df.columns and 'now_cost' in full_df.columns:
	  full_df['bonus_per_cost'] = safe_divide('bonus', 'now_cost', full_df)
	if 'bonus' in full_df.columns and 'minutes' in full_df.columns:
		full_df['bonus_per_minute'] = safe_divide('bonus', 'minutes', full_df)
	if 'bps' in full_df.columns and 'now_cost' in full_df.columns:
		full_df['bps_per_cost'] = safe_divide('bps', 'now_cost', full_df)
	if 'bps' in full_df.columns and 'minutes' in full_df.columns:
		full_df['bps_per_minute'] = safe_divide('bps', 'minutes', full_df)
	if 'expected_goals' in full_df.columns and 'now_cost' in full_df.columns:
		full_df['expected_goals_per_cost'] = safe_divide('expected_goals', 'now_cost', full_df)
	if 'expected_goals' in full_df.columns and 'minutes' in full_df.columns:
		full_df['expected_goals_per_minute'] = safe_divide('expected_goals', 'minutes', full_df)
	if 'expected_assists' in full_df.columns and 'now_cost' in full_df.columns:
		full_df['expected_assists_per_cost'] = safe_divide('expected_assists', 'now_cost', full_df)
	if 'expected_assists' in full_df.columns and 'minutes' in full_df.columns:
		full_df['expected_assists_per_minute'] = safe_divide('expected_assists', 'minutes', full_df)
	if 'expected_goal_involvements' in full_df.columns and 'now_cost' in full_df.columns:
		full_df['expected_goal_involvements_per_cost'] = safe_divide('expected_goal_involvements', 'now_cost', full_df)
	if 'expected_goal_involvements' in full_df.columns and 'minutes' in full_df.columns:
		full_df['expected_goal_involvements_per_minute'] = safe_divide('expected_goal_involvements', 'minutes', full_df)
	if 'goals_scored' in full_df.columns and 'expected_goals' in full_df.columns:
		full_df["goals_over_expected"] = np.abs(full_df["goals_scored"] - full_df["expected_goals"])
	if 'assists' in full_df.columns and 'expected_assists' in full_df.columns:
		full_df["assists_over_expected"] = np.abs(full_df["assists"] - full_df["expected_assists"])
	if 'goals_scored' in full_df.columns and 'now_cost' in full_df.columns:
		full_df['goals_scored_per_cost'] = safe_divide('goals_scored', 'now_cost', full_df)
	if 'goals_scored' in full_df.columns and 'minutes' in full_df.columns:
		full_df['goals_scored_per_minute'] = safe_divide('goals_scored', 'minutes', full_df)
	if 'assists' in full_df.columns and 'now_cost' in full_df.columns:
		full_df['assists_per_cost'] = safe_divide('assists', 'now_cost', full_df)
	if 'assists' in full_df.columns and 'minutes' in full_df.columns:
		full_df['assists_per_minute'] = safe_divide('assists', 'minutes', full_df)


	# List of features to adjust (make sure these columns exist after processing)
	features_to_adjust = [
		("minutes", False), ("total_points", False), ("points_per_cost", False),
		("points_per_minute", False), ("bonus", False), ("bonus_per_cost", False),
		("bonus_per_minute", False), ("bps", False), ("bps_per_cost", False),
		("bps_per_minute", False), ("expected_goals", False),
		("expected_goals_per_cost", False), ("expected_goals_per_minute", False),
		("expected_assists", False), ("expected_assists_per_cost", False),
		("expected_assists_per_minute", False), ("expected_goal_involvements", False),
		("expected_goal_involvements_per_cost", False),
		("expected_goal_involvements_per_minute", False),
		("goals_scored", False), ("goals_scored_per_cost", False),
		("goals_scored_per_minute", False), ("assists", False),
		("assists_per_cost", False), ("assists_per_minute", False),
		("goals_over_expected", True), ("assists_over_expected", True)
	]

	adj_cols = []
	for feature, is_reversed in features_to_adjust:
		if feature in full_df.columns:
			full_df = create_adjusted_feature(full_df, feature, reversed=is_reversed)
			adj_cols.append(f"{feature}_adj")
		else:
			print(f"Warning: Feature '{feature}' not found for adjustment.")

	# Create Final Value
	if adj_cols: # Only sum if there are adjusted columns
		full_df['Final_value'] = full_df[adj_cols].sum(axis=1)
	else:
		print("Warning: No adjusted features were created. Setting Final_value to 0.")
		full_df['Final_value'] = 0.0 # Assign a default value

	# Keep essential columns for selection and display
	final_cols = ["web_name", "position", "team", "now_cost", "total_points", "Final_value"] # Add others if needed for display
	# Ensure all columns in final_cols actually exist in full_df before selecting
	final_cols = [col for col in final_cols if col in full_df.columns]
	if not final_cols:
		print("Error: None of the essential final columns exist. Cannot proceed.")
		return None
	full_df = full_df[final_cols].copy()


	pd.options.display.float_format = "{:,.2f}".format # Keep for potential debugging output

	print("Data processing complete.")
	return full_df

# ==============================================================================
def run_team_selection(total_budget=100.0, sub_factor=0.2):
	"""
	Main function to fetch data, process it, and select the team.
	Returns the first team, subs, and captain DataFrames.
	"""
	raw_fpl_data = get_data_fpl()
	if raw_fpl_data is None:
		return None, None, None # Indicate failure

	processed_data = process_data(raw_fpl_data)
	if processed_data is None or processed_data.empty:
		print("Error: Data processing failed or resulted in empty DataFrame.")
		return None, None, None # Indicate failure

	if 'Final_value' not in processed_data.columns or processed_data['Final_value'].isnull().all():
		print("Error: 'Final_value' column is missing or all NaN after processing.")
		return None, None, None

	if 'now_cost' not in processed_data.columns:
		print("Error: 'now_cost' column is missing after processing.")
		return None, None, None


	# Check if there are enough players per position after filtering
	min_players_check = processed_data['position'].value_counts()
	required_min = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
	sufficient_players = True
	for pos, count in required_min.items():
		if min_players_check.get(pos, 0) < count:
			print(f"Warning: Insufficient players for position {pos} after filtering ({min_players_check.get(pos, 0)} < {count}). Team selection might fail or be suboptimal.")
			sufficient_players = False
			# Depending on requirements, you might want to return None here if strict player numbers are needed

	# Proceed with selection even with warnings, but the optimization might not find a valid solution
	print(f"Running team selection with Budget: {total_budget}, Sub Factor: {sub_factor}")
	try:
		first_team, subs, captain = select_team(processed_data, sub_factor=sub_factor, total_budget=total_budget)
		print("Team selection complete.")
		return first_team, subs, captain
	except Exception as e: # Catch potential errors during PuLP solve if constraints are infeasible
		print(f"Error during team selection optimization: {e}")
		return None, None, None


# ==============================================================================
# Remove the old script execution block:
# df = get_data_fpl(...)
# ... processing steps ...
# ... printing GKP, DEF, MID, FWD ...
# ... select_team call ...
# ... printing final team ...
# sys.exit()
# ==============================================================================