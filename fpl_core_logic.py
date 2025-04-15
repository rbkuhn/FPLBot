"""Core logic for fetching FPL data, processing it, and selecting a team."""

import requests
import numpy as np
import pandas as pd
import pulp
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Constants ---
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
TEAMS = {
    1: "Arsenal", 2: "Villa", 3: "Bournemouth", 4: "Brentford", 5: "Brighton",
    6: "Chelsea", 7: "Palace", 8: "Everton", 9: "Fullham", 10: "Ipswich",
    11: "Leicester", 12: "Liverpool", 13: "Man City", 14: "Man United",
    15: "Newcastle", 16: "Forest", 17: "Southampton", 18: "Spurs",
    19: "West Ham", 20: "Wolves"
}
# Columns required from the FPL API data after initial processing
REQUIRED_COLS = [
    "web_name", "position", "team", "now_cost", "minutes",
    "total_points", "bonus", "bps", "selected_by_percent",
    "goals_scored", "assists", "expected_goals", "expected_assists",
    "expected_goal_involvements", "influence", "creativity",
    "threat", "form", "ep_this", "ep_next", "transfers_in_event",
    "transfers_out_event", "status", "element_type" # Keep status/element_type for processing
]
# Map user-friendly weight names to the base feature names
# (These base names correspond to the first element in FEATURES_TO_ADJUST tuples)
FEATURE_WEIGHT_MAP = {
    'points': ["total_points", "points_per_minute", "bonus", "bonus_per_minute", "bps", "bps_per_minute"],
    'value': ["points_per_cost", "bonus_per_cost", "bps_per_cost"],
    'form': ["form"], # Add form_per_cost, form_per_minute if created
    'expected_goals': ["expected_goals", "expected_goals_per_cost", "expected_goals_per_minute", "goals_over_expected"],
    'expected_assists': ["expected_assists", "expected_assists_per_cost", "expected_assists_per_minute", "assists_over_expected"],
    'fixtures': ["avg_fdr_next_5"], # Added fixtures mapping
    # Add more mappings as needed, e.g., for involvements, threat, creativity, etc.
}
# Features to engineer and scale for the final player value calculation.
# Tuple format: (feature_name, should_higher_be_better?)
# Note: For 'over_expected' stats, lower absolute difference might be seen as better (closer to expectation),
# hence reversed=True (meaning lower values get higher scaled scores).
FEATURES_TO_ADJUST = [
    ("minutes", True), ("total_points", True), ("points_per_cost", True),
    ("points_per_minute", True), ("bonus", True), ("bonus_per_cost", True),
    ("bonus_per_minute", True), ("bps", True), ("bps_per_cost", True),
    ("bps_per_minute", True), ("expected_goals", True),
    ("expected_goals_per_cost", True), ("expected_goals_per_minute", True),
    ("expected_assists", True), ("expected_assists_per_cost", True),
    ("expected_assists_per_minute", True), ("expected_goal_involvements", True),
    ("expected_goal_involvements_per_cost", True),
    ("expected_goal_involvements_per_minute", True),
    ("goals_scored", True), ("goals_scored_per_cost", True),
    ("goals_scored_per_minute", True), ("assists", True),
    ("assists_per_cost", True), ("assists_per_minute", True),
    ("goals_over_expected", False), # Lower abs difference is better -> reverse scale
    ("assists_over_expected", False), # Lower abs difference is better -> reverse scale
    ("avg_fdr_next_5", False), # Add FDR, lower is better (False)
]


# --- Data Fetching ---

def get_data_fpl(url=FPL_API_URL):
    """Fetches the main bootstrap static data from the FPL API.

    Args:
        url (str): The URL for the FPL bootstrap-static endpoint.

    Returns:
        pd.DataFrame: A DataFrame containing player and team data,
                      or None if fetching fails.
    """
    logging.info("Getting official FPL stats from %s ...", url)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        json_data = response.json()
        logging.info("Successfully fetched FPL bootstrap data.")
        return json_data
    except requests.exceptions.RequestException as e:
        logging.error("Error fetching FPL data: %s", e)
        return None
    except KeyError as e:
        logging.error("Error parsing FPL data: Missing key %s", e)
        return None

def get_fixture_data(url=FPL_FIXTURES_URL):
    """Fetches future fixture data from the FPL API.

    Args:
        url (str): The URL for the FPL fixtures endpoint.

    Returns:
        pd.DataFrame: A DataFrame containing fixture information,
                      or None if fetching fails.
    """
    logging.info("Getting FPL fixture data from %s ...", url)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        fixtures_data = response.json()
        logging.info("Successfully fetched FPL fixture data.")
        return pd.DataFrame(fixtures_data)
    except requests.exceptions.RequestException as e:
        logging.error("Error fetching FPL fixture data: %s", e)
        return None
    except ValueError as e: # Catches JSON decoding errors
        logging.error("Error parsing FPL fixture data (JSON): %s", e)
        return None


# --- Data Processing ---

def _safe_divide(numerator_col, denominator_col, df_in):
    """Helper function for safe division, handling NaNs and zeros in denominator."""
    num = df_in[numerator_col]
    den = df_in[denominator_col]
    # Replace 0s and NaNs in denominator with NaN, then perform division, then fill resulting NaNs with 0
    return (num / den.replace(0, np.nan)).fillna(0)


def create_adjusted_feature(df, feature_name, higher_is_better=True):
    """Scales a feature within each position group (0 to 1) based on min/max.

    Creates a new column named '{feature_name}_adj'.

    Args:
        df (pd.DataFrame): DataFrame containing player data including 'position'
                           and the feature column.
        feature_name (str): The name of the column to scale.
        higher_is_better (bool): If True, scales normally (max = 1).
                                 If False, scales reversed (min = 1).

    Returns:
        pd.DataFrame: The DataFrame with the new adjusted feature column added.
    """
    adj_col_name = f"{feature_name}_adj"
    df[adj_col_name] = 0.0  # Initialize column

    for pos in POSITIONS.values():
        pos_mask = df["position"] == pos
        if not pos_mask.any():
            continue  # Skip if no players in this position

        feature_series = df.loc[pos_mask, feature_name]
        min_val = feature_series.min()
        max_val = feature_series.max()

        # Handle edge cases: NaN range, max=0, or all values the same
        if pd.isna(min_val) or pd.isna(max_val) or max_val == 0:
            continue
        if min_val == max_val:
            # If all values are the same, scaling depends on the direction
            # Normal scale: all get 1.0 if max != 0 (already handled), else 0
            # Reversed scale: all get 1.0
            df.loc[pos_mask, adj_col_name] = 1.0 if not higher_is_better else (1.0 if max_val != 0 else 0.0)
            continue

        # Apply scaling
        if higher_is_better:
            # Normal scaling (0 to 1, higher is better)
            df.loc[pos_mask, adj_col_name] = (feature_series - min_val) / (max_val - min_val)
        else:
            # Reversed scaling (0 to 1, lower is better)
            df.loc[pos_mask, adj_col_name] = 1.0 - (feature_series - min_val) / (max_val - min_val)

    # Ensure any NaNs possibly introduced (though unlikely with checks) are filled
    df[adj_col_name] = df[adj_col_name].fillna(0)
    return df


def process_data(raw_df, fixture_df=None, next_gameweek_id=None, min_minutes_played=0, feature_weights=None):
    """Processes the raw FPL data to prepare it for team selection.

    Args:
        raw_df (pd.DataFrame): Raw DataFrame from get_data_fpl ('elements').
        fixture_df (pd.DataFrame, optional): Fixture DataFrame from get_fixture_data.
        next_gameweek_id (int, optional): ID of the next gameweek.
        min_minutes_played (int): Minimum minutes player must have played.
        feature_weights (list, optional): User-selected feature categories.

    Returns:
        pd.DataFrame: Processed DataFrame ready for optimization,
                      or None if processing fails.
    """
    if raw_df is None or raw_df.empty:
        logging.warning("Input DataFrame is empty or None. Skipping processing.")
        return None

    logging.info("Processing FPL data...")
    df = raw_df.copy()

    # --- Define Raw Columns Needed ---
    # These are columns expected directly from the API before transformation
    raw_required_cols = [
        "web_name", "team", "element_type", "now_cost", "minutes",
        "total_points", "bonus", "bps", "selected_by_percent",
        "goals_scored", "assists", "expected_goals", "expected_assists",
        "expected_goal_involvements", "influence", "creativity",
        "threat", "form", "ep_this", "ep_next", "transfers_in_event",
        "transfers_out_event", "status"
    ]

    # Check for required RAW columns before proceeding
    missing_cols = [col for col in raw_required_cols if col not in df.columns]
    if missing_cols:
        logging.error("Missing required columns in raw FPL data: %s", missing_cols)
        return None

    # --- Pre-process Fixture Data ---
    team_upcoming_fdr = {}
    num_fixtures_to_consider = 5 # Hardcoded for now, make parameter later
    if fixture_df is not None and not fixture_df.empty and next_gameweek_id is not None:
        logging.info(f"Processing fixtures for next {num_fixtures_to_consider} gameweeks starting from GW{next_gameweek_id}.")
        try:
            # Ensure necessary columns exist
            fixture_cols = ['event', 'team_h', 'team_a', 'team_h_difficulty', 'team_a_difficulty']
            if not all(col in fixture_df.columns for col in fixture_cols):
                logging.warning("Fixture data missing required columns. Skipping FDR calculation.")
            else:
                # Filter for upcoming gameweeks
                upcoming_fixtures = fixture_df[
                    (fixture_df['event'] >= next_gameweek_id) &
                    (fixture_df['event'] < next_gameweek_id + num_fixtures_to_consider)
                ].copy()

                # Create a lookup: team_id -> list of FDR scores
                for _, row in upcoming_fixtures.iterrows():
                    h_team = row['team_h']
                    a_team = row['team_a']
                    h_fdr = row['team_h_difficulty']
                    a_fdr = row['team_a_difficulty']

                    if h_team not in team_upcoming_fdr: team_upcoming_fdr[h_team] = []
                    if a_team not in team_upcoming_fdr: team_upcoming_fdr[a_team] = []

                    team_upcoming_fdr[h_team].append(a_fdr) # Home team faces away team difficulty
                    team_upcoming_fdr[a_team].append(h_fdr) # Away team faces home team difficulty

        except Exception as e:
            logging.error(f"Error processing fixture data: {e}", exc_info=True)
            team_upcoming_fdr = {} # Reset on error
    else:
        logging.warning("Fixture data or next gameweek ID not available. Skipping FDR calculation.")

    # --- Initial Filtering and Mapping ---
    # Keep raw team id needed for FDR lookup later
    df['raw_team_id'] = df['team']
    df['team'] = df['team'].map(TEAMS).fillna('Unknown Team')
    df['position'] = df['element_type'].map(POSITIONS).fillna('Unknown Pos')
    # Filter status
    initial_count = len(df)
    df = df[df['status'] == 'a'].copy()
    logging.info(f"Filtered {initial_count - len(df)} unavailable players.")
    # Filter minutes
    initial_count = len(df)
    df = df[df['minutes'] >= min_minutes_played].copy()
    logging.info(
        f"Filtered {initial_count - len(df)} players with < {min_minutes_played} minutes."
    )
    logging.info(f"Players remaining for selection: {len(df)}")
    if df.empty:
        logging.warning("No players remaining after filtering. Cannot proceed.")
        return None

    # --- Column Selection and Type Conversion ---
    # Select columns needed for processing first
    # Include raw_team_id temporarily for FDR calc
    cols_for_processing = [col for col in REQUIRED_COLS if col in df.columns] + ['raw_team_id']
    if 'position' not in cols_for_processing:
        logging.error("Critical error: 'position' column not created during mapping.")
        return None
    df = df[list(set(cols_for_processing))].copy() # Use set to avoid duplicate raw_team_id if error

    numeric_cols = [
        col for col in df.columns
        if col not in ["web_name", "position", "team", "status", "element_type", "raw_team_id"]
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True)
    if 'now_cost' in df.columns:
        df['now_cost'] = df['now_cost'] / 10.0

    # --- Calculate Average FDR per Player --- (Moved After Col Selection/Typing)
    num_fixtures_to_consider = 5 # Keep consistent
    fdr_col_name = f'avg_fdr_next_{num_fixtures_to_consider}'
    df[fdr_col_name] = np.nan # Initialize column

    if team_upcoming_fdr:
        # Ensure raw_team_id exists before using it
        if 'raw_team_id' in df.columns:
            for index, row in df.iterrows():
                team_id = row['raw_team_id']
                if team_id in team_upcoming_fdr:
                    upcoming_fdr_list = team_upcoming_fdr[team_id]
                    if upcoming_fdr_list:
                        avg_fdr = sum(upcoming_fdr_list) / len(upcoming_fdr_list)
                        df.loc[index, fdr_col_name] = avg_fdr
                # else: Leave as NaN if team not in fixture lookup

            # Fill NaNs after calculation
            median_fdr = df[fdr_col_name].median()
            fill_value = median_fdr if pd.notna(median_fdr) else 5.0
            df[fdr_col_name] = df[fdr_col_name].fillna(fill_value)
            logging.info(f"Calculated average FDR for next {num_fixtures_to_consider} games.")
        else:
            logging.warning("'raw_team_id' column missing after selection. Skipping FDR calculation.")
            df[fdr_col_name] = df[fdr_col_name].fillna(3.0) # Neutral value
    else:
        # FDR calculation was skipped earlier
        df[fdr_col_name] = df[fdr_col_name].fillna(3.0) # Neutral value

    # Drop raw_team_id after use
    df.drop(columns=['raw_team_id'], inplace=True, errors='ignore')

    # --- Feature Engineering ---
    logging.info("Engineering features...")
    if 'total_points' in df.columns and 'now_cost' in df.columns:
        df['points_per_cost'] = _safe_divide('total_points', 'now_cost', df)
    if 'total_points' in df.columns and 'minutes' in df.columns:
        df['points_per_minute'] = _safe_divide('total_points', 'minutes', df)
    if 'bonus' in df.columns and 'now_cost' in df.columns:
        df['bonus_per_cost'] = _safe_divide('bonus', 'now_cost', df)
    if 'bonus' in df.columns and 'minutes' in df.columns:
        df['bonus_per_minute'] = _safe_divide('bonus', 'minutes', df)
    if 'bps' in df.columns and 'now_cost' in df.columns:
        df['bps_per_cost'] = _safe_divide('bps', 'now_cost', df)
    if 'bps' in df.columns and 'minutes' in df.columns:
        df['bps_per_minute'] = _safe_divide('bps', 'minutes', df)
    if 'expected_goals' in df.columns and 'now_cost' in df.columns:
        df['expected_goals_per_cost'] = _safe_divide('expected_goals', 'now_cost', df)
    if 'expected_goals' in df.columns and 'minutes' in df.columns:
        df['expected_goals_per_minute'] = _safe_divide('expected_goals', 'minutes', df)
    if 'expected_assists' in df.columns and 'now_cost' in df.columns:
        df['expected_assists_per_cost'] = _safe_divide('expected_assists', 'now_cost', df)
    if 'expected_assists' in df.columns and 'minutes' in df.columns:
        df['expected_assists_per_minute'] = _safe_divide('expected_assists', 'minutes', df)
    if 'expected_goal_involvements' in df.columns and 'now_cost' in df.columns:
        df['expected_goal_involvements_per_cost'] = _safe_divide('expected_goal_involvements', 'now_cost', df)
    if 'expected_goal_involvements' in df.columns and 'minutes' in df.columns:
        df['expected_goal_involvements_per_minute'] = _safe_divide('expected_goal_involvements', 'minutes', df)
    if 'goals_scored' in df.columns and 'now_cost' in df.columns:
        df['goals_scored_per_cost'] = _safe_divide('goals_scored', 'now_cost', df)
    if 'goals_scored' in df.columns and 'minutes' in df.columns:
        df['goals_scored_per_minute'] = _safe_divide('goals_scored', 'minutes', df)
    if 'assists' in df.columns and 'now_cost' in df.columns:
        df['assists_per_cost'] = _safe_divide('assists', 'now_cost', df)
    if 'assists' in df.columns and 'minutes' in df.columns:
        df['assists_per_minute'] = _safe_divide('assists', 'minutes', df)

    # Calculate absolute difference between actual and expected goals/assists
    if 'goals_scored' in df.columns and 'expected_goals' in df.columns:
        df["goals_over_expected"] = np.abs(df["goals_scored"] - df["expected_goals"])
    if 'assists' in df.columns and 'expected_assists' in df.columns:
        df["assists_over_expected"] = np.abs(df["assists"] - df["expected_assists"])

    # --- Feature Scaling and Final Value Calculation ---
    logging.info("Scaling features (including FDR)...")
    all_adj_feature_cols = []
    for feature, high_is_good in FEATURES_TO_ADJUST:
        if feature in df.columns:
            df = create_adjusted_feature(df, feature, higher_is_better=high_is_good)
            all_adj_feature_cols.append(f"{feature}_adj")
            logging.debug(f"Scaled feature: {feature}")
        else:
            # Log warning if base feature (incl. FDR) not found before scaling
            logging.warning("Feature '%s' not found for scaling adjustment.", feature)

    # --- Determine Columns for Final Value based on Weights ---
    final_value_cols = []
    if feature_weights:
        logging.info(f"Using feature weights: {feature_weights}")
        selected_base_features = set() # Use a set to avoid double counting
        for weight_key in feature_weights:
            if weight_key in FEATURE_WEIGHT_MAP:
                selected_base_features.update(FEATURE_WEIGHT_MAP[weight_key])
            else:
                logging.warning(f"Unknown feature weight key: '{weight_key}'")

        # Convert selected base features to their adjusted column names
        for base_feature in selected_base_features:
            adj_col = f"{base_feature}_adj"
            if adj_col in all_adj_feature_cols:
                final_value_cols.append(adj_col)
            else:
                # This might happen if the base feature wasn't in the original df
                logging.debug(f"Adjusted column '{adj_col}' for weighted feature '{base_feature}' not available.")

        # Always include minutes played for basic filtering relevance
        # (Alternative: make 'minutes' its own weight category)
        minutes_adj_col = "minutes_adj"
        if minutes_adj_col in all_adj_feature_cols and minutes_adj_col not in final_value_cols:
            final_value_cols.append(minutes_adj_col)
            logging.info("Including 'minutes_adj' in Final Value calculation.")

        if not final_value_cols:
            logging.warning("No adjusted features selected based on weights. Defaulting to all available.")
            final_value_cols = all_adj_feature_cols # Fallback to all if selection results in none
    else:
        logging.info("No feature weights specified, using all available adjusted features.")
        final_value_cols = all_adj_feature_cols # Default to all if no weights provided

    # --- Final Value Calculation ---
    if final_value_cols:
        df['Final_value'] = df[final_value_cols].sum(axis=1)
        logging.info("Calculated 'Final_value' based on %d features: %s", len(final_value_cols), ", ".join(final_value_cols))
    else:
        logging.warning("No adjusted features available to calculate Final_value. Setting to 0.")
        df['Final_value'] = 0.0

    # --- Final Column Selection ---
    # Ensure the new avg_fdr column is available for display if needed
    final_cols_to_keep = [
        "web_name", "position", "team", "now_cost", "total_points", "Final_value",
        f"avg_fdr_next_{num_fixtures_to_consider}" # Add the calculated FDR column
    ]
    final_cols = [col for col in final_cols_to_keep if col in df.columns]
    if 'Final_value' not in final_cols:
        logging.error("'Final_value' column missing before final selection.")
        return None # Or handle differently if FDR is critical
    if not final_cols:
        logging.error("None of the essential final columns exist after processing.")
        return None

    processed_df = df[final_cols].copy()
    logging.info("Data processing complete.")
    return processed_df


# --- Team Selection ---

def select_team(processed_df, sub_factor=0.2, total_budget=100.0, captain_positions=None, formation='any', feature_weights=None):
    """Selects the optimal FPL team using linear programming.

    Args:
        processed_df (pd.DataFrame): DataFrame processed by process_data().
                                     Must contain 'Final_value', 'now_cost',
                                     'position', and 'team'.
        sub_factor (float): Factor (0-1) applied to the 'Final_value' of
                            substitutes in the objective function.
        total_budget (float): The maximum budget allowed for the 15 players.
        captain_positions (list, optional): List of positions allowed for captain (e.g., ['MID', 'FWD']). Defaults to ['MID'].
        formation (str, optional): Desired formation (e.g., '3-4-3', '4-4-2', 'any'). Defaults to 'any'.
        feature_weights (list, optional): Feature categories to include in 'Final_value'. Defaults to all.

    Returns:
        tuple: Contains three DataFrames (first_team_df, subs_df, captain_df)
               sorted by position. Returns (None, None, None) if optimization fails
               or if input data is invalid.
    """
    # Default captain positions if none provided
    if captain_positions is None:
        captain_positions = ['MID']

    if processed_df is None or processed_df.empty:
        logging.error("Cannot select team: Processed DataFrame is empty or None.")
        return None, None, None
    required_cols_opt = ["Final_value", "now_cost", "position", "team"]
    if not all(col in processed_df.columns for col in required_cols_opt):
        logging.error("Processed DataFrame missing required columns for optimization: %s",
                      [c for c in required_cols_opt if c not in processed_df.columns])
        return None, None, None

    num_players = len(processed_df)
    if num_players < 15:
        logging.warning("Warning: Fewer than 15 players available (%d). May not find a valid squad.", num_players)
        # Allow to proceed, but PuLP might fail

    logging.info("Setting up team selection optimization problem...")
    model = pulp.LpProblem('FPL_Team_Selection', pulp.LpMaximize)

    # Decision variables: 1 if player i is in starting 11, 0 otherwise
    decisions = [
        pulp.LpVariable(f"x{i}", lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    # Decision variables: 1 if player i is captain, 0 otherwise
    captain_decisions = [
        pulp.LpVariable(f"y{i}", lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    # Decision variables: 1 if player i is a substitute, 0 otherwise
    sub_decisions = [
        pulp.LpVariable(f"z{i}", lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]

    # --- Objective Function ---
    # Maximize total 'Final_value' of starters + captain bonus + scaled value of subs
    model += pulp.lpSum(
        (captain_decisions[i] + decisions[i] + sub_decisions[i] * sub_factor) * processed_df.iloc[i]['Final_value']
        for i in range(num_players)
    ), "Total_Objective_Value"

    # --- Constraints ---
    # Budget constraint
    model += pulp.lpSum(
        (decisions[i] + sub_decisions[i]) * processed_df.iloc[i]['now_cost']
        for i in range(num_players)
    ) <= total_budget, "Total_Cost"

    # Squad size constraints
    model += pulp.lpSum(decisions) == 11, "Total_Starters"
    model += pulp.lpSum(sub_decisions) == 4, "Total_Subs"

    # Position constraints (for the 15-player squad)
    pos_indices = {pos: [i for i, p in enumerate(processed_df['position']) if p == pos] for pos in POSITIONS.values()}
    model += pulp.lpSum(decisions[i] + sub_decisions[i] for i in pos_indices["GKP"]) == 2, "Goalkeepers"
    model += pulp.lpSum(decisions[i] + sub_decisions[i] for i in pos_indices["DEF"]) == 5, "Defenders"
    model += pulp.lpSum(decisions[i] + sub_decisions[i] for i in pos_indices["MID"]) == 5, "Midfielders"
    model += pulp.lpSum(decisions[i] + sub_decisions[i] for i in pos_indices["FWD"]) == 3, "Forwards"

    # Starting lineup formation constraints
    model += pulp.lpSum(decisions[i] for i in pos_indices["GKP"]) == 1, "Starting_Goalkeeper"

    # -- Apply specific formation OR default min/max constraints --
    if formation != 'any' and len(formation) == 3 and formation.isdigit():
        try:
            num_def = int(formation[0])
            num_mid = int(formation[1])
            num_fwd = int(formation[2])
            logging.info(f"Applying formation constraint: {num_def}-{num_mid}-{num_fwd}")

            # Validate formation sum (excluding GKP)
            if num_def + num_mid + num_fwd != 10:
                logging.warning(f"Invalid formation sum ({num_def}+{num_mid}+{num_fwd} != 10). Using default constraints.")
                # Apply default min/max if formation is invalid
                model += pulp.lpSum(decisions[i] for i in pos_indices["DEF"]) >= 3, "Min_Starting_Defenders"
                model += pulp.lpSum(decisions[i] for i in pos_indices["DEF"]) <= 5, "Max_Starting_Defenders"
                model += pulp.lpSum(decisions[i] for i in pos_indices["MID"]) >= 3, "Min_Starting_Midfielders"
                model += pulp.lpSum(decisions[i] for i in pos_indices["MID"]) <= 5, "Max_Starting_Midfielders"
                model += pulp.lpSum(decisions[i] for i in pos_indices["FWD"]) >= 1, "Min_Starting_Forwards"
                model += pulp.lpSum(decisions[i] for i in pos_indices["FWD"]) <= 3, "Max_Starting_Forwards"
            else:
                # Apply exact formation constraints
                model += pulp.lpSum(decisions[i] for i in pos_indices["DEF"]) == num_def, f"Formation_{num_def}_DEF"
                model += pulp.lpSum(decisions[i] for i in pos_indices["MID"]) == num_mid, f"Formation_{num_mid}_MID"
                model += pulp.lpSum(decisions[i] for i in pos_indices["FWD"]) == num_fwd, f"Formation_{num_fwd}_FWD"

        except (ValueError, IndexError):
            logging.warning(f"Could not parse formation string '{formation}'. Using default constraints.")
            # Apply default min/max if parsing fails
            model += pulp.lpSum(decisions[i] for i in pos_indices["DEF"]) >= 3, "Min_Starting_Defenders"
            model += pulp.lpSum(decisions[i] for i in pos_indices["DEF"]) <= 5, "Max_Starting_Defenders"
            model += pulp.lpSum(decisions[i] for i in pos_indices["MID"]) >= 3, "Min_Starting_Midfielders"
            model += pulp.lpSum(decisions[i] for i in pos_indices["MID"]) <= 5, "Max_Starting_Midfielders"
            model += pulp.lpSum(decisions[i] for i in pos_indices["FWD"]) >= 1, "Min_Starting_Forwards"
            model += pulp.lpSum(decisions[i] for i in pos_indices["FWD"]) <= 3, "Max_Starting_Forwards"
    else:
        # Apply default min/max if formation is 'any' or invalid format
        if formation != 'any':
             logging.warning(f"Invalid formation format '{formation}'. Using default constraints.")
        model += pulp.lpSum(decisions[i] for i in pos_indices["DEF"]) >= 3, "Min_Starting_Defenders"
        model += pulp.lpSum(decisions[i] for i in pos_indices["DEF"]) <= 5, "Max_Starting_Defenders"
        model += pulp.lpSum(decisions[i] for i in pos_indices["MID"]) >= 3, "Min_Starting_Midfielders"
        model += pulp.lpSum(decisions[i] for i in pos_indices["MID"]) <= 5, "Max_Starting_Midfielders"
        model += pulp.lpSum(decisions[i] for i in pos_indices["FWD"]) >= 1, "Min_Starting_Forwards"
        model += pulp.lpSum(decisions[i] for i in pos_indices["FWD"]) <= 3, "Max_Starting_Forwards"

    # Club constraint (max 3 players per team in the 15-player squad)
    for team_name in processed_df.team.unique():
        team_indices = [i for i, t in enumerate(processed_df['team']) if t == team_name]
        model += pulp.lpSum(
            decisions[i] + sub_decisions[i] for i in team_indices
        ) <= 3, f"Max_3_Players_{team_name.replace(' ', '_')}"

    # Captain constraints
    model += pulp.lpSum(captain_decisions) == 1, "Single_Captain"
    # Ensure captain is a starter (Constraint: captain_decisions[i] <= decisions[i])
    # This is handled implicitly by the objective function and the fact that captains add value,
    # but also explicitly linked later.

    # Apply captain position constraints based on user selection
    if captain_positions:
        captain_pos_indices = []
        for pos in captain_positions:
            if pos in pos_indices:
                captain_pos_indices.extend(pos_indices[pos])
            else:
                logging.warning(f"Position '{pos}' selected for captaincy not found in player data.")

        if captain_pos_indices:
            # Constraint: The sum of captain variables for players in allowed positions must equal 1
            # (Since total captains must be 1, this forces the captain to be from this group)
            model += pulp.lpSum(
                captain_decisions[i] for i in captain_pos_indices
            ) == 1, "Captain_Position_Constraint"
            logging.info(f"Applied captain position constraint: Captain must be one of {', '.join(captain_positions)}")
        else:
            logging.warning("No players found matching the selected captain positions. Captain constraint not applied.")
            # If no players match, the model might become infeasible if it strictly requires a captain.
            # The Single_Captain constraint still exists, so PuLP will likely fail unless this logic is changed.
    else:
        logging.warning("No captain positions specified. Captain can be any player.")
        # If no positions are specified, we don't add the position constraint.
        # The model will pick the best captain based on the objective function.

    # Linking constraints (Ensure captain starts, player is starter OR sub)
    for i in range(num_players):
        model += (decisions[i] - captain_decisions[i]) >= 0, f"Captain_Must_Start_{i}"
        model += (decisions[i] + sub_decisions[i]) <= 1, f"Player_Starts_Or_Sub_{i}"

    # --- Solve ---
    logging.info("Solving optimization problem...")
    try:
        # Use a specific solver if needed, e.g., pulp.PULP_CBC_CMD(msg=0)
        status = model.solve()
        logging.info("Solver status: %s", pulp.LpStatus[status])

        if pulp.LpStatus[status] != 'Optimal':
            logging.warning("Optimal solution not found. Status: %s", pulp.LpStatus[status])
            # Optionally, try relaxing constraints or analyze the model if infeasible
            return None, None, None

    except Exception as e:
        logging.error("Error during PuLP solve: %s", e)
        return None, None, None

    # --- Extract Results ---
    logging.info("Extracting selected team...")
    player_indices = [i for i in range(num_players) if decisions[i].varValue > 0.9] # Use varValue for float results
    sub_player_indices = [i for i in range(num_players) if sub_decisions[i].varValue > 0.9]
    cap_player_indices = [i for i in range(num_players) if captain_decisions[i].varValue > 0.9]

    # Basic validation
    if len(player_indices) != 11 or len(sub_player_indices) != 4 or len(cap_player_indices) != 1:
         logging.error("Error: Optimization result has incorrect number of players/captain.")
         logging.error(f"Starters: {len(player_indices)}, Subs: {len(sub_player_indices)}, Captain: {len(cap_player_indices)}")
         # Consider returning the partial results or None
         return None, None, None

    first_team_df = processed_df.iloc[player_indices].sort_values(by="position", ascending=False)
    subs_df = processed_df.iloc[sub_player_indices].sort_values(by="position", ascending=False)
    captain_df = processed_df.iloc[cap_player_indices] # Captain is usually just one player

    logging.info("Team selection complete.")
    return first_team_df, subs_df, captain_df


# --- Main Orchestration ---

def run_team_selection(total_budget=100.0, sub_factor=0.2, min_minutes=0, captain_positions=None, feature_weights=None, formation='any'):
    """Main function to fetch data, process it, and select the team.

    Args:
        total_budget (float): Maximum budget for the squad.
        sub_factor (float): Weighting factor for substitutes' value.
        min_minutes (int): Minimum minutes played filter for players.
        captain_positions (list): List of positions allowed for captain.
        feature_weights (list): List of features categories to weight.
        formation (str): Preferred formation (e.g., '343', '442', 'any').

    Returns:
        tuple: Contains three DataFrames (first_team, subs, captain) or
               (None, None, None) if any step fails.
    """
    # Fetch base data (now returns full JSON)
    bootstrap_data = get_data_fpl()
    if bootstrap_data is None:
        logging.error("Failed to fetch FPL bootstrap data. Cannot proceed.")
        return None, None, None

    # Extract player data
    try:
        raw_player_data = pd.DataFrame(bootstrap_data['elements'])
    except KeyError:
        logging.error("Could not find 'elements' key in bootstrap data.")
        return None, None, None

    # Find the next gameweek ID
    next_gameweek_id = None
    try:
        all_events = bootstrap_data.get('events', [])
        for event in all_events:
            if event.get('is_next') is True:
                next_gameweek_id = event.get('id')
                break
        if next_gameweek_id is None:
            logging.warning("Could not determine the next gameweek from bootstrap data.")
            # Handle this case - maybe default to GW1 or fail?
            # For now, let's try to proceed but FDR calculation will likely fail.
        else:
            logging.info(f"Next gameweek identified: {next_gameweek_id}")
    except Exception as e:
        logging.error(f"Error processing events to find next gameweek: {e}")
        # Proceed without next_gameweek_id, FDR calculation will likely fail.

    # Fetch fixture data
    fixture_df = get_fixture_data()
    # Proceed even if fixture_df is None, processing function should handle it

    # Pass feature_weights down to process_data
    processed_data = process_data(
        raw_player_data, # Use extracted player data
        fixture_df=fixture_df, # Pass fixture data
        next_gameweek_id=next_gameweek_id, # Pass next gameweek ID
        min_minutes_played=min_minutes,
        feature_weights=feature_weights
    )
    if processed_data is None or processed_data.empty:
        logging.error("Data processing failed or resulted in empty DataFrame.")
        return None, None, None

    # Check for sufficient players per position before optimization attempt
    min_players_check = processed_data['position'].value_counts()
    # Required for a valid 15-man squad structure
    squad_min = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    sufficient_players = True
    for pos, count in squad_min.items():
        available_count = min_players_check.get(pos, 0)
        if available_count < count:
            logging.warning(
                f"Insufficient players for position {pos} after filtering "
                f"({available_count} available < {count} required). "
                f"Team selection might fail or be suboptimal."
            )
            sufficient_players = False
            # Decide if you want to stop here if strict numbers are needed
            # return None, None, None

    if not sufficient_players:
        logging.warning("Proceeding with selection despite insufficient players in some positions.")

    # Run the optimization
    first_team, subs, captain = select_team(
        processed_data,
        sub_factor=sub_factor,
        total_budget=total_budget,
        captain_positions=captain_positions,
        formation=formation,
        feature_weights=feature_weights # Pass weights to select_team too if needed later?
                                        # For now, weights only affect process_data
    )

    # select_team logs its own errors if it returns None
    return first_team, subs, captain


# --- Example Usage (for testing script directly) ---
if __name__ == '__main__':
    logging.info("--- Running FPL Core Logic Directly (Test Mode) ---")
    test_budget = 100.0
    test_sub_factor = 0.2
    test_min_minutes = 90 * 5 # Example: filter for players with at least 5 games worth of minutes

    ft, sb, cap = run_team_selection(
        total_budget=test_budget,
        sub_factor=test_sub_factor,
        min_minutes=test_min_minutes
    )

    if ft is not None and sb is not None and cap is not None:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        print("\n--- Selected First Team ---")
        print(ft)
        print("\n--- Selected Substitutes ---")
        print(sb)
        print("\n--- Selected Captain ---")
        print(cap)

        total_cost = round(ft['now_cost'].sum() + sb['now_cost'].sum(), 2)
        total_points = ft['total_points'].sum() # Summing points of starters only
        captain_points_bonus = cap['total_points'].iloc[0] # Add captain points once more
        total_display_points = total_points + captain_points_bonus

        print("\n--- Summary ---")
        print(f"Total Cost: {total_cost:.1f}m")
        print(f"Total Points (Starters + Captain): {total_display_points}") # This is season total, not predicted
    else:
        print("\n--- Team selection failed. Check logs above for errors. ---")
    logging.info("--- FPL Core Logic Direct Run Finished ---")