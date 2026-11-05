# Flight Delay Prediction System

Flask backend plus a plain HTML/CSS/JS frontend for the Unit 2–5 project.

## Run
    pip install -r requirements.txt
    # optional, enables /api/predict: copy the Kaggle flights.csv to backend/data/flights.csv
    python backend/app.py
Open http://127.0.0.1:5000

## API
- GET  /api/health   status and whether the predictor is trained
- GET  /api/results  recorded results from Units 2-5
- POST /api/predict  JSON: month, day, day_of_week, hour, distance. Returns an estimated arrival delay (minutes)

The predictor is the Unit 2 least-squares model (R2 about 0.01), so treat its output as a demo, not a forecast.
