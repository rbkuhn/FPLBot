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
    *   Displays key stats like xG, xA, xGi, and % Ownership.
    *   Basic user feedback for input errors and selection failures (Flash messages).

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