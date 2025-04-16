from flask import Flask, render_template, request, redirect, url_for, flash
# Import the function from your refactored script
from fpl_core_logic import run_team_selection
import logging

app = Flask(__name__)
# A secret key is needed for session management (used by flash messages)
# In production, use a strong, environment-variable-based key
app.secret_key = 'dev_secret_key' # Replace with a real secret key for production

@app.route('/')
def index():
    # Later, we'll add logic here to get data and pass it to the template
    return render_template('index.html')

@app.route('/select_team', methods=['POST'])
def select_fpl_team():
    try:
        # Get basic parameters
        budget = float(request.form['budget'])
        sub_factor = float(request.form['sub_factor'])
        min_minutes = int(request.form.get('min_minutes', 0))
        captain_positions = request.form.getlist('captain_positions')
        if not captain_positions:
            flash("Warning: No captain position selected, defaulting to Midfielder.", category='warning')
            captain_positions = ['MID']
        feature_weights = request.form.getlist('feature_weights')
        if not feature_weights:
            flash("Warning: No feature weights selected, using default balanced approach.", category='warning')
            # feature_weights = list(FEATURE_WEIGHT_MAP.keys()) # Needs import or definition
            # Let's handle the default logic more robustly later if needed, for now just flash.
            pass # Avoid error if FEATURE_WEIGHT_MAP is not accessible here
        formation = request.form.get('formation', 'any')
        # Get differential weighting flag (checkbox value is 'true' if checked, or None)
        use_differential = request.form.get('differential_weighting') == 'true'

        # Add specific validation
        if budget < 50 or budget > 200:
             flash("Error: Budget must be between 50.0 and 200.0", category='error')
             return redirect(url_for('index'))
        if sub_factor < 0 or sub_factor > 1:
             flash("Error: Substitution Factor must be between 0.0 and 1.0", category='error')
             return redirect(url_for('index'))
        if min_minutes < 0:
            flash("Error: Minimum Minutes Played cannot be negative.", category='error')
            return redirect(url_for('index'))
        if not captain_positions:
             flash("Error: At least one captain position must be selected.", category='error')
             return redirect(url_for('index'))

    except ValueError:
        flash("Error: Invalid input. Budget, Sub Factor, and Minutes must be numbers.", category='error')
        return redirect(url_for('index'))
    except KeyError as e:
        flash(f"Error: Missing required form field: {e}", category='error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"An unexpected error occurred processing the form: {str(e)}", category='error')
        logging.error(f"Unexpected form processing error: {e}", exc_info=True)
        return redirect(url_for('index'))

    logging.info(f"Received request: Budget={budget}, Sub Factor={sub_factor}, Min Minutes={min_minutes}")
    logging.info(f"Captain Positions: {captain_positions}, Formation: {formation}")
    logging.info(f"Feature Weights: {feature_weights}")
    logging.info(f"Use Differential Weighting: {use_differential}")

    # Call the main function from your script with the new parameters
    try:
        first_team, subs, captain = run_team_selection(
            total_budget=budget,
            sub_factor=sub_factor,
            min_minutes=min_minutes,
            captain_positions=captain_positions,
            feature_weights=feature_weights,
            formation=formation,
            use_differential=use_differential
        )
    except Exception as e:
        # Catch potential unexpected errors from the core logic
        logging.error(f"Unexpected error during run_team_selection: {e}", exc_info=True)
        flash("An unexpected error occurred during team selection. Please check logs.", category='error')
        return redirect(url_for('index'))

    if first_team is None or subs is None or captain is None:
        # Handle expected errors during team selection (logged in core logic)
        flash("Error: Team selection failed. This might be due to overly strict constraints (e.g., budget too low, invalid formation, not enough players available after filtering) or an API issue. Try adjusting parameters.", category='error')
        return redirect(url_for('index'))

    # --- Calculate Total Historical Points (BEFORE renaming) ---
    total_historical_points = 0
    if first_team is not None and captain is not None and not first_team.empty and not captain.empty:
        if 'total_points' in first_team.columns and 'total_points' in captain.columns:
            try:
                starters_points = first_team['total_points'].sum()
                captain_points = captain['total_points'].iloc[0]
                total_historical_points = starters_points + captain_points
                logging.info(f"Calculated total historical points: Starters={starters_points}, Captain={captain_points}, Total={total_historical_points}")
            except (KeyError, IndexError, TypeError) as e:
                logging.error(f"Error calculating historical points: {e}")
                total_historical_points = "Error"
        else:
            logging.warning("'total_points' column missing. Cannot calculate historical points.")
            total_historical_points = "N/A"
    else:
        total_historical_points = "N/A"

    # --- Reorder columns (using original names) ---
    target_col = 'Final_value'
    if first_team is not None and target_col in first_team.columns:
        cols = [col for col in first_team.columns if col != target_col] + [target_col]
        first_team = first_team[cols]
    # No need to reorder subs/captain if display order matches first_team target

    # --- Rename columns for display ---
    column_rename_map = {
        "web_name": "Name",
        "position": "Pos",
        "team": "Team",
        "now_cost": "Cost (£m)",
        "total_points": "Points",
        "minutes": "Mins",
        "form": "Form",
        "bonus": "Bonus",
        "bps": "BPS",
        "expected_goals": "xG",
        "expected_assists": "xA",
        "expected_goal_involvements": "xGi",
        "selected_by_percent": "% Ownership",
        "avg_fdr_next_5": "Avg FDR",
        "Final_value": "Score"
    }
    if first_team is not None:
        first_team = first_team.rename(columns=column_rename_map)
    if subs is not None:
        subs = subs.rename(columns=column_rename_map)
    if captain is not None:
        captain = captain.rename(columns=column_rename_map)

    # --- Get Captain Name (AFTER renaming) ---
    captain_name = None
    if captain is not None and not captain.empty and "Name" in captain.columns:
        try:
            captain_name = captain["Name"].iloc[0]
            logging.info(f"Identified Captain: {captain_name}")
        except (IndexError, KeyError) as e:
            logging.error(f"Could not extract captain name: {e}")

    # --- Tooltip Map (for template) ---
    tooltip_map = {
        "Pos": "Position (GKP=Goalkeeper, DEF=Defender, MID=Midfielder, FWD=Forward)",
        "Cost (£m)": "Current Price in Millions",
        "Points": "Total FPL Points Scored This Season",
        "Mins": "Total Minutes Played This Season",
        "Form": "Player's FPL Form Rating",
        "Bonus": "Total Bonus Points Scored This Season",
        "BPS": "Total Bonus Points System Score This Season",
        "xG": "Expected Goals (Based on chance quality)",
        "xA": "Expected Assists (Based on chance creation quality)",
        "xGi": "Expected Goal Involvement (xG + xA)",
        "% Ownership": "Percentage of FPL Managers Owning This Player",
        "Avg FDR": "Average Fixture Difficulty Rating of Next 5 Games",
        "Score": "Calculated Player Value Score (Based on Selected Weights and Differentials)"
    }

    # --- Calculate Total Cost (using RENAMED column) ---
    total_cost = 0.0
    if first_team is not None and 'Cost (£m)' in first_team.columns:
        total_cost += first_team['Cost (£m)'].sum()
    if subs is not None and 'Cost (£m)' in subs.columns:
        total_cost += subs['Cost (£m)'].sum()
    total_cost = round(total_cost, 2)

    # --- Prepare data for template ---
    first_team_list = first_team.to_dict(orient='records') if first_team is not None else []
    subs_list = subs.to_dict(orient='records') if subs is not None else []
    headers = list(first_team.columns) if first_team is not None and not first_team.empty else [] # Use columns from renamed DF

    return render_template('team.html',
                           headers=headers, # Pass headers
                           tooltip_map=tooltip_map, # Pass tooltips
                           first_team_list=first_team_list, # Pass list of dicts
                           subs_list=subs_list, # Pass list of dicts
                           captain_name=captain_name, # Pass captain's name
                           total_cost=total_cost,
                           total_historical_points=total_historical_points,
                           budget=budget,
                           sub_factor=sub_factor,
                           min_minutes=min_minutes,
                           captain_positions=captain_positions,
                           feature_weights=feature_weights,
                           formation=formation,
                           use_differential=use_differential)

# Add more routes later for team selection, etc.

if __name__ == '__main__':
    app.run(debug=True) # debug=True allows auto-reloading during development