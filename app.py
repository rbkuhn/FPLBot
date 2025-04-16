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

    # --- Reorder columns to move 'Final_value' to the end ---
    target_col = 'Final_value'
    if first_team is not None and target_col in first_team.columns:
        cols = [col for col in first_team.columns if col != target_col] + [target_col]
        first_team = first_team[cols]
    if subs is not None and target_col in subs.columns:
        cols = [col for col in subs.columns if col != target_col] + [target_col]
        subs = subs[cols]
    if captain is not None and target_col in captain.columns:
        cols = [col for col in captain.columns if col != target_col] + [target_col]
        captain = captain[cols]
    # --- End Reorder ---

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
        "avg_fdr_next_5": "Avg FDR", # Assuming this is the fixture column name
        "Final_value": "Score"
        # Add other columns if they appear and need renaming
    }

    if first_team is not None:
        first_team = first_team.rename(columns=column_rename_map)
    if subs is not None:
        subs = subs.rename(columns=column_rename_map)
    if captain is not None:
        captain = captain.rename(columns=column_rename_map)
    # --- End Rename ---

    # Convert dataframes to HTML tables for easy rendering in the template
    first_team_html = first_team.to_html(classes='table table-striped', index=False, border=0)
    subs_html = subs.to_html(classes='table table-striped', index=False, border=0)
    captain_html = captain.to_html(classes='table table-striped', index=False, border=0)
    # Use the RENAMED column name to calculate total cost
    total_cost = round(first_team['Cost (£m)'].sum() + subs['Cost (£m)'].sum(), 2)

    return render_template('team.html',
                           first_team_table=first_team_html,
                           subs_table=subs_html,
                           captain_table=captain_html,
                           total_cost=total_cost,
                           # Pass parameters to display on results page
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