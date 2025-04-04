from flask import Flask, render_template, request, redirect, url_for
# Import the function from your refactored script
from fpl_core_logic import run_team_selection

app = Flask(__name__)

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

        # Get minimum minutes played
        min_minutes = int(request.form.get('min_minutes', 0))

        # Get captain position preferences
        captain_positions = request.form.getlist('captain_positions')
        if not captain_positions:  # Default to MID if none selected
            captain_positions = ['MID']

        # Get feature weights
        feature_weights = request.form.getlist('feature_weights')
        if not feature_weights:  # Default to all if none selected
            feature_weights = ['points', 'value', 'form', 'expected_goals', 'expected_assists']

        # Get formation preference
        formation = request.form.get('formation', 'any')

    except (ValueError, KeyError) as e:
        # Handle cases where form data is missing or not a number
        print(f"Error: Invalid form input - {str(e)}")
        return redirect(url_for('index')) # Simple redirect for now

    print(f"Received request: Budget={budget}, Sub Factor={sub_factor}, Min Minutes={min_minutes}")
    print(f"Captain Positions: {captain_positions}, Formation: {formation}")
    print(f"Feature Weights: {feature_weights}")

    # Call the main function from your script with the new parameters
    first_team, subs, captain = run_team_selection(
        total_budget=budget,
        sub_factor=sub_factor,
        min_minutes=min_minutes,
        captain_positions=captain_positions,
        feature_weights=feature_weights,
        formation=formation
    )

    if first_team is None or subs is None or captain is None:
        # Handle errors during team selection (e.g., data fetch failed, processing error, infeasible solution)
        # Redirect back to form with an error message (implementation TBD)
        print("Error: Team selection failed.")
        # Optionally pass an error message to the index page via flash messages or query params
        return redirect(url_for('index'))

    # Convert dataframes to HTML tables for easy rendering in the template
    # The `to_html` method includes default styling which we might override later with CSS
    first_team_html = first_team.to_html(classes='table table-striped', index=False, border=0)
    subs_html = subs.to_html(classes='table table-striped', index=False, border=0)
    captain_html = captain.to_html(classes='table table-striped', index=False, border=0)

    # Calculate total cost
    total_cost = round(first_team['now_cost'].sum() + subs['now_cost'].sum(), 2)

    # Pass the HTML tables and total cost to the result template
    return render_template('team.html',
                           first_team_table=first_team_html,
                           subs_table=subs_html,
                           captain_table=captain_html,
                           total_cost=total_cost)

# Add more routes later for team selection, etc.

if __name__ == '__main__':
    app.run(debug=True) # debug=True allows auto-reloading during development