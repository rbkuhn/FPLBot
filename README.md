# FPL AutoBot

A Flask web application to help select an optimal Fantasy Premier League (FPL) team based on current player statistics and user-defined constraints.

## Features

*   Fetches the latest player data from the official FPL API.
*   Processes player statistics to calculate various metrics (points per cost, points per minute, expected goals/assists adjusted values, etc.).
*   Uses linear programming (PuLP) to select an optimal 15-player squad (11 starters, 4 substitutes).
*   **Configurable Selection Criteria:**
    *   Total budget.
    *   Substitution factor (how much to value bench players).
    *   Minimum minutes played filter.
    *   Preferred formation (e.g., 3-4-3, 4-4-2) or allow any valid formation.
    *   Allowed positions for captain selection.
    *   Weighting of different feature categories (points, value, form, expected stats) in the player valuation (`Final_value`).
*   Adheres to FPL constraints:
    *   Budget limit.
    *   Squad composition (2 GKP, 5 DEF, 5 MID, 3 FWD).
    *   Starting lineup formation rules (dynamic based on preference or standard rules).
    *   Maximum 3 players per club.
    *   Selects a Captain from allowed positions.
*   Web interface to input criteria and view the selected team.
*   Basic user feedback for input errors and selection failures (Flash messages).

## Technology Stack

*   **Backend:** Python, Flask
*   **Data Handling:** Pandas, NumPy
*   **Optimization:** PuLP
*   **FPL API Interaction:** Requests
*   **Frontend:** HTML, CSS (Basic)
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
    ```

## Usage

1.  **Run the Flask application:**
    ```bash
    python app.py
    ```
2.  **Open your web browser** and navigate to `http://127.0.0.1:5000` (or the address provided in the terminal).
3.  **Enter your desired criteria** using the form (Budget, Sub Factor, Min Minutes, Captain Positions, Feature Weights, Formation Preference).
4.  **Click "Select My Team"**.
5.  The application will fetch data, perform the optimization based on your inputs, and display the selected First Team, Substitutes, Captain, total cost, and the parameters used. Error messages will appear on the form page if issues occur.

## Notes

*   Data fetching and processing can take a few seconds.
*   The 'Final Value' metric is now calculated based on the selected feature weights. See `FEATURE_WEIGHT_MAP` and the `Final_value` calculation in `fpl_core_logic.py` for details.
*   The PuLP solver might not find an optimal solution if the constraints (budget, formation, available players after filtering) are too tight.

## Future Enhancements

*   **Fixture Difficulty:** Incorporate upcoming Fixture Difficulty Ratings (FDR) into player valuation or filtering.
*   ~~**Form Weighting:** Allow users to adjust the weighting between long-term stats and recent form.~~ (Implemented via Feature Weighting)
*   **Differential Value:** Add an option to boost the value of low-ownership players (`selected_by_percent`).
*   **BGW/DGW Handling:** Add features to help plan for blank and double gameweeks (e.g., excluding players with blanks).
*   **Injury/Status Display:** Show player status (injured, doubtful) and news more prominently in the results.
*   ~~**Advanced Captaincy Logic:** Implement different strategies for captain selection (e.g., highest `ep_next`, best fixture).~~ (Partially implemented via position constraints)
*   **Underlying Stats Visibility:** Display key expected stats (xG, xA) in the output tables.
*   **UI/UX Improvements:**
    *   ~~Enhance user feedback (flash messages for errors).~~ (Implemented)
    *   Add table sorting.
    *   Add tooltips for form inputs/stats.
    *   Add loading indicator. 