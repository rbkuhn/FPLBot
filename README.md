# FPL AutoBot

A Flask web application to help select an optimal Fantasy Premier League (FPL) team based on current player statistics and constraints.

## Features

*   Fetches the latest player data from the official FPL API.
*   Processes player statistics to calculate various metrics (points per cost, points per minute, expected goals/assists adjusted values, etc.).
*   Uses linear programming (PuLP) to select an optimal 15-player squad (11 starters, 4 substitutes) based on maximizing a calculated 'Final Value' metric.
*   Adheres to FPL constraints:
    *   Budget limit (configurable).
    *   Squad composition (2 GKP, 5 DEF, 5 MID, 3 FWD).
    *   Starting lineup formation rules (1 GKP, 3-5 DEF, 3-5 MID, 1-3 FWD).
    *   Maximum 3 players per club.
    *   Selects a Captain (currently constrained to be a Midfielder).
*   Simple web interface to input budget and substitution factor, and view the selected team.

## Technology Stack

*   **Backend:** Python, Flask
*   **Data Handling:** Pandas, NumPy
*   **Optimization:** PuLP
*   **FPL API Interaction:** Requests
*   **Frontend:** HTML, CSS (Basic)

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
3.  **Enter your desired total budget** (e.g., `100.0`) and **substitution factor** (a value between 0 and 1, representing how much the substitutes' 'Final Value' contributes to the optimization objective, e.g., `0.2`).
4.  **Click "Select My Team"**.
5.  The application will fetch data, perform the optimization, and display the selected First Team, Substitutes, Captain, and the total cost.

## Notes

*   Data fetching and processing can take a few seconds.
*   The 'Final Value' metric is a sum of various adjusted performance stats. The specific components can be seen and modified in `FPL_AutoBot.py`.
*   The PuLP solver might not find an optimal solution if the constraints (especially budget) are too tight or if there aren't enough available players meeting the criteria after filtering.

## Future Enhancements

*   **Fixture Difficulty:** Incorporate upcoming Fixture Difficulty Ratings (FDR) into player valuation or filtering.
*   **Form Weighting:** Allow users to adjust the weighting between long-term stats and recent form.
*   **Differential Value:** Add an option to boost the value of low-ownership players (`selected_by_percent`).
*   **BGW/DGW Handling:** Add features to help plan for blank and double gameweeks (e.g., excluding players with blanks).
*   **Injury/Status Display:** Show player status (injured, doubtful) and news more prominently in the results.
*   **Advanced Captaincy Logic:** Implement different strategies for captain selection (e.g., highest `ep_next`, best fixture).
*   **Underlying Stats Visibility:** Display key expected stats (xG, xA) in the output tables.
*   **UI/UX Improvements:** Enhance user feedback (flash messages for errors), add table sorting, tooltips, etc. 