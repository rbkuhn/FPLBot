# FPL AutoBot

A Flask web application to help select an optimal Fantasy Premier League (FPL) team based on current player statistics and user-defined constraints.

## Features

*   Fetches the latest player data and fixture information from the official FPL API.
*   Processes player statistics to calculate various metrics (points per cost, points per minute, expected goals/assists adjusted values, average upcoming fixture difficulty, etc.).
*   Uses linear programming (PuLP) to select an optimal 15-player squad (11 starters, 4 substitutes).
*   **Configurable Selection Criteria:**
    *   Total budget.
    *   Substitution factor (how much to value bench players).
    *   Minimum minutes played filter.
    *   Preferred formation (e.g., 3-4-3, 4-4-2), correctly enforced, or allows any valid formation.
    *   Allowed positions for captain selection.
    *   Weighting of different feature categories (points, value, form, expected stats, fixtures) in the player valuation (`Final_value`).
        *   *Note:* Currently, selecting 'fixtures' treats higher raw FDR values as contributing positively to the score.
    *   Optional differential weighting (boosts value for low-ownership players).
*   Adheres to FPL constraints:
    *   Budget limit.
    *   Squad composition (2 GKP, 5 DEF, 5 MID, 3 FWD).
    *   Starting lineup formation rules (applies standard rules or the specific valid formation requested).
    *   Maximum 3 players per club.
    *   Selects a Captain from allowed positions.
*   **Web Interface (UI/UX Enhancements):**
    *   Input criteria and view the selected team.
    *   Custom background image.
    *   Wider page layout for better table visibility.
    *   Semi-transparent backgrounds on text elements for improved readability over the background image.
    *   User-friendly column headers in result tables (e.g., "Cost (£m)", "Points", "xG", "xGi", "% Ownership", "Score").
    *   Tooltips on table headers explaining the metric.
    *   Score column formatted to 3 decimal places.
    *   Highlights the captain with bold text, a background color, and "(C)" marker in the First Team list.
    *   Displays total historical points for the selected starting 11 (including double captain points).
    *   Displays key stats like xG, xA, xGi, and % Ownership.
    *   Basic user feedback for input errors and selection failures (Flash messages).
    *   Selection Parameters section moved to the bottom of the results page for better layout flow.

## Technology Stack

*   **Backend:** Python, Flask
*   **Data Handling:** Pandas, NumPy
*   **Optimization:** PuLP
*   **FPL API Interaction:** Requests
*   **Frontend:** HTML, CSS
*   **Logging:** Python `logging` module

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rbkuhn/FPLBot.git # Replace with your repo URL if different
    cd FPLBot
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # You might also need autopep8 for code formatting
    # pip install autopep8
    ```

## Usage

1.  **Ensure `background.jpg` is present** in the `static/` directory.
2.  **Run the Flask application:**
    ```bash
    python app.py
    ```
3.  **Open your web browser** and navigate to `http://127.0.0.1:5000` (or the address provided in the terminal).
4.  **Enter your desired criteria** using the form (Budget, Sub Factor, Min Minutes, Captain Positions, Feature Weights, Formation Preference).
5.  **Click "Select My Team"**.
6.  The application will fetch data, perform the optimization based on your inputs (correctly applying the chosen formation), and display the selected First Team, Substitutes, Captain, total cost, parameters used, and various player stats with readable headers.

## Notes

*   Data fetching and processing can take a few seconds.
*   The 'Score' column (`Final_value` internally) is calculated based on the selected feature weights. See `FEATURE_WEIGHT_MAP` and the `Final_value` calculation in `fpl_core_logic.py` for details.
*   The PuLP solver might not find an optimal solution if the constraints (budget, formation, available players after filtering) are too tight.

## Future Enhancements

*   **Transfer Suggestions:** Allow users to input their FPL Team ID, fetch their current squad, and suggest optimal transfers (out/in) based on the app's player valuation logic.
*   **BGW/DGW Handling:** Add features to help plan for blank and double gameweeks (e.g., excluding players with blanks).
*   **Injury/Status Display:** Show player status (injured, doubtful) and news more prominently in the results.
*   **UI/UX Improvements:**
    *   Add table sorting.
    *   Add tooltips for form inputs/stats.
    *   Add loading indicator.
    *   Add user control for number of fixtures to consider (currently hardcoded).
*   **Advanced Analytics:**
    *   **Player Form Analysis:** Implement rolling averages for player performance metrics over the last 3-6 gameweeks.
    *   **Team Strength Analysis:** Calculate team-specific attacking/defending strength metrics based on recent performances.
    *   **Fixture Difficulty Rating (FDR) Enhancement:** Consider home/away performance and recent form when calculating FDR.
    *   **Expected Points (xP) Model:** Develop a comprehensive expected points model incorporating xG, xA, clean sheet probability, and bonus point potential.
*   **Team Management Features:**
    *   **Chip Strategy Planner:** Help users plan when to use their chips (Wildcard, Free Hit, Bench Boost, Triple Captain).
    *   **Price Change Predictor:** Track and predict player price changes based on transfer activity.
    *   **Team Value Tracker:** Monitor and optimize team value over time.
    *   **Auto-Substitution Logic:** Implement smart bench ordering based on fixture difficulty and player form.
*   **League Analysis:**
    *   **Mini-League Analysis:** Compare team performance against mini-league rivals.
    *   **Template Team Detection:** Identify common player picks in top teams.
    *   **Differential Finder:** Highlight high-value players with low ownership.
*   **Data Visualization:**
    *   **Player Performance Charts:** Visualize player performance trends over time.
    *   **Fixture Calendar:** Interactive calendar view of upcoming fixtures with difficulty ratings.
    *   **Team Strength Matrix:** Visual representation of team attacking/defending strengths.
*   **Machine Learning Integration:**
    *   **Player Performance Prediction:** Use historical data to predict future performance.
    *   **Optimal Captain Selection:** ML-based captain selection considering form, fixtures, and historical performance.
    *   **Transfer Success Predictor:** Predict the success rate of potential transfers.
*   **API Enhancements:**
    *   **Real-time Updates:** Implement WebSocket connections for live updates.
    *   **Historical Data Analysis:** Access and analyze historical FPL data for better predictions.
    *   **Player Comparison Tool:** Compare multiple players across various metrics.
*   **User Experience:**
    *   **Custom Alerts:** Set up notifications for price changes, injuries, and team news.
    *   **Mobile Responsive Design:** Optimize the interface for mobile devices.
    *   **Dark Mode:** Add a dark mode option for the interface.
    *   **Export Functionality:** Allow exporting team data to CSV/Excel for further analysis.
*   **Social Features:**
    *   **Team Sharing:** Share team selections and strategies with other users.
    *   **Community Ratings:** Allow users to rate and comment on player picks.
    *   **Expert Insights:** Integrate expert opinions and analysis from FPL content creators.
*   **Gameweek Planning:**
    *   **Multi-Gameweek Planner:** Plan transfers and team selections across multiple gameweeks.
    *   **Rotation Planner:** Optimize team selection considering fixture congestion.
    *   **Captain Rotation Strategy:** Plan captain choices across multiple gameweeks.
*   **Performance Metrics:**
    *   **Team Value vs Points Analysis:** Track the relationship between team value and points scored.
    *   **Transfer Success Rate:** Analyze the success rate of transfers made.
    *   **Bench Points Analysis:** Track and optimize bench points contribution.
*   **Integration Features:**
    *   **FPL API Integration:** Direct integration with FPL API for automatic team updates.
    *   **Fantasy Football Scout Integration:** Access additional player statistics and analysis.
    *   **Twitter Integration:** Share team updates and receive news alerts.