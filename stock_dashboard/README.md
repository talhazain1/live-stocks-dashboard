# Live Stocks Dashboard

A real-time and historical data dashboard for stocks and options, powered by Polygon.io and Dash.

## Features

- Historical OHLC charts with SMA, EMA, and RSI indicators
- Live trade price updates (via v3 API)
- Option quotes and snapshots
- Option minute bars and time & sales for any stock (with fallback to 15-min delayed aggregates)
- Pre-configured panels for NVDA, universal options, and raw trade data

## Requirements

- Python 3.7+
- A free Polygon.io Developer API key

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/talhazain1/live-stocks-dashboard.git
   cd live-stocks-dashboard
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file inside the `stock_dashboard` folder:
   ```bash
   cp stock_dashboard/.env.example stock_dashboard/.env
   ```
   Add your Polygon API key:
   ```ini
   POLYGON_API_KEY=YOUR_POLYGON_API_KEY_HERE
   ```

5. Run the app locally:
   ```bash
   python stock_dashboard/app.py
   ```

## Deployment

This project is ready for deployment on Heroku, Render, or any PaaS that supports Python and Gunicorn.

- **Heroku**:
  ```bash
  heroku create
  git push heroku main
  heroku config:set POLYGON_API_KEY=YOUR_POLYGON_API_KEY_HERE
  ```

- **Render**:
  - Create a new Web Service.
  - Connect your GitHub repo.
  - Set **build command** to `pip install -r requirements.txt`.
  - Set **start command** to `gunicorn stock_dashboard.app:server`.
  - Add an environment variable:
    ```bash
    POLYGON_API_KEY=YOUR_POLYGON_API_KEY_HERE
    ```

## License

MIT 