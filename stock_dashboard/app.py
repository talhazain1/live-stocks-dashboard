import os
from datetime import datetime, timedelta
import requests
import pandas as pd
from dotenv import load_dotenv
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import builtins

# Override the built-in print function
builtins.print = lambda *args, **kwargs: None

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
API_KEY = os.getenv("POLYGON_API_KEY", "SCesdecLGSpkJCTZ1AUepRMlhr5vxnq1")

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Polygon Stock Dashboard")
server = app.server

def fetch_historical(ticker, start_date, end_date):
    """
    Fetch historical daily OHLC data for the given ticker between start_date and end_date.
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {"apiKey": API_KEY}
    resp = requests.get(url, params=params)
    data = resp.json()
    if "results" in data:
        df = pd.DataFrame(data["results"])
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        return df
    return pd.DataFrame()

# Fetch time & sales (intraday trades) for a given ticker
def fetch_time_sales(ticker, date=None, limit=50):
    """
    Fetch time and sales tick data for the given ticker on specified date.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/ticks/stocks/trades/{ticker}/{date}"
    params = {"apiKey": API_KEY, "limit": limit, "sort": "desc"}
    # Debug: log request
    resp = requests.get(url, params=params)
    print(f"[TimeSales] GET {url} params={params} status={resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        print(f"[TimeSales] Response not valid JSON: {resp.text}")
        return pd.DataFrame(columns=["time", "price", "size"])
    # Debug: log response keys and sample
    print(f"[TimeSales] Response keys: {list(data.keys())}")
    if "results" in data:
        print(f"[TimeSales] Retrieved {len(data['results'])} ticks")
        ts_df = pd.DataFrame(data["results"])
        # Determine timestamp column
        if "t" in ts_df.columns:
            ts_df["time"] = pd.to_datetime(ts_df["t"], unit="us")
        elif "sip_timestamp" in ts_df.columns:
            ts_df["time"] = pd.to_datetime(ts_df["sip_timestamp"], unit="ms")
        elif "timestamp" in ts_df.columns:
            ts_df["time"] = pd.to_datetime(ts_df["timestamp"], unit="ms")
        # Rename price and size columns if needed
        ts_df = ts_df.rename(columns={"p": "price", "s": "size"}) if "p" in ts_df.columns else ts_df
        ts_df = ts_df[["time", "price", "size"]]
        return ts_df
    # No results returned
    print(f"[TimeSales] No 'results' in response, returning empty DataFrame")
    return pd.DataFrame(columns=["time", "price", "size"])

# Fetch latest option trade
def fetch_option_last_trade(ticker):
    """
    Fetch the most recent trade for a given option ticker using v3 endpoint.
    """
    # Clean ticker (remove O: prefix if present)
    clean_ticker = ticker.upper().strip()
    if clean_ticker.startswith("O:"):
        clean_ticker = clean_ticker.split(":", 1)[1]
    url = f"https://api.polygon.io/v3/trades/{clean_ticker}"
    params = {"apiKey": API_KEY, "limit": 1, "sort": "timestamp", "order": "desc"}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"[OptionTrade] HTTP {resp.status_code} for {clean_ticker}: {resp.text}")
        return {}
    data = resp.json()
    results = data.get("results", [])
    return results[0] if results else {}

# Fetch latest option quote
def fetch_option_last_quote(ticker):
    """
    Fetch the most recent quote for a given option ticker using v3 endpoint.
    """
    clean_ticker = ticker.upper().strip()
    if clean_ticker.startswith("O:"):
        clean_ticker = clean_ticker.split(":", 1)[1]
    url = f"https://api.polygon.io/v3/quotes/{clean_ticker}"
    params = {"apiKey": API_KEY, "limit": 1, "sort": "timestamp", "order": "desc"}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"[OptionQuote] HTTP {resp.status_code} for {clean_ticker}: {resp.text}")
        return {}
    data = resp.json()
    results = data.get("results", [])
    return results[0] if results else {}

# Fetch minute aggregate bars for a given option ticker
def fetch_option_minute_bars(ticker, start_date, end_date):
    """
    Fetch minute-level aggregate OHLC bars for the given option ticker between start_date and end_date.
    """
    # Ensure option ticker is prefixed with O:
    clean_ticker = ticker.upper().strip()
    if not clean_ticker.startswith("O:"):
        clean_ticker = f"O:{clean_ticker}"
    url = f"https://api.polygon.io/v2/aggs/ticker/{clean_ticker}/range/1/minute/{start_date}/{end_date}"
    params = {"apiKey": API_KEY}
    # Debug: log request
    resp = requests.get(url, params=params)
    print(f"[OptionMinuteBars] GET {url} params={params} status={resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        print(f"[OptionMinuteBars] JSON decode error for {clean_ticker}: {resp.text}")
        return pd.DataFrame()
    # Debug: log response keys
    print(f"[OptionMinuteBars] Response keys: {list(data.keys())}")
    if "results" in data:
        print(f"[OptionMinuteBars] Retrieved {len(data['results'])} bars")
        df = pd.DataFrame(data["results"])
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        return df
    # No results
    print(f"[OptionMinuteBars] No 'results' in response, message: {data.get('message')}")
    return pd.DataFrame()

# Fetch list of option contracts for a given underlying
def fetch_option_contracts(underlying, expired=False, limit=1000):
    """Fetch active option contracts for the underlying ticker."""
    url = "https://api.polygon.io/v3/reference/options/contracts"
    params = {"apiKey": API_KEY, "underlying_ticker": underlying, "expired": expired, "limit": limit}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"[OptionContracts] HTTP {resp.status_code} for {underlying}: {resp.text}")
        return []
    data = resp.json()
    return data.get("results", [])

# Fetch snapshot for a given option contract
def fetch_option_snapshot(underlying, contract):
    """
    Fetch snapshot (last trade, last quote, greeks, etc.) for a given options contract.
    """
    url = f"https://api.polygon.io/v3/snapshot/options/{underlying}/{contract}"
    params = {"apiKey": API_KEY}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"[OptionSnapshot] HTTP {resp.status_code} for {underlying}/{contract}: {resp.text}")
        return {}
    data = resp.json()
    return data.get("results", {})

# Fetch time & sales for a given option contract (v3 trades)
def fetch_option_trades(contract, limit=50, cursor=None, sort="desc"):
    """
    Fetch time and sales tick data for the given option contract using v3 endpoint.
    """
    clean_ticker = contract.upper().strip()
    if clean_ticker.startswith("O:"):
        clean_ticker = clean_ticker.split(":", 1)[1]
    url = f"https://api.polygon.io/v3/trades/{clean_ticker}"
    params = {"apiKey": API_KEY, "limit": limit, "sort": "timestamp", "order": sort}
    if cursor:
        params["cursor"] = cursor
    resp = requests.get(url, params=params)
    try:
        data = resp.json()
    except ValueError:
        print(f"[OptionTrades] Response not valid JSON: {resp.text}")
        return pd.DataFrame(columns=["time", "price", "size"])
    results = data.get("results", [])
    if not results:
        return pd.DataFrame(columns=["time", "price", "size"])
    df = pd.DataFrame(results)
    # Determine timestamp column
    if "sip_timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["sip_timestamp"], unit="ns")
    elif "participant_timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["participant_timestamp"], unit="ns")
    elif "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[["time", "price", "size"]]

# Fetch 1-min aggregate bars for a stock ticker between start_date and end_date
def fetch_stock_minute_bars(ticker, start_date, end_date):
    clean = ticker.upper().strip()
    url = f"https://api.polygon.io/v2/aggs/ticker/{clean}/range/1/minute/{start_date}/{end_date}"
    params = {"apiKey": API_KEY}
    resp = requests.get(url, params=params)
    try:
        data = resp.json()
    except ValueError:
        return pd.DataFrame()
    if "results" in data:
        df = pd.DataFrame(data["results"])
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        return df
    return pd.DataFrame()

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("🚀 Polygon Stock Dashboard", className="text-center text-primary mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Ticker Symbol"),
            dcc.Dropdown(
                id="ticker-dropdown",
                options=[
                    {"label": "Apple (AAPL)", "value": "AAPL"},
                    {"label": "Microsoft (MSFT)", "value": "MSFT"},
                    {"label": "Google (GOOG)", "value": "GOOG"},
                    {"label": "Amazon (AMZN)", "value": "AMZN"},
                    {"label": "Tesla (TSLA)", "value": "TSLA"},
                    {"label": "NVIDIA (NVDA)", "value": "NVDA"}
                ],
                value="AAPL",
                clearable=False,
                className="form-control"
            )
        ], width=2),
        dbc.Col([
            html.Label("Date Range"),
            dcc.DatePickerRange(
                id="date-picker",
                start_date=(datetime.now() - timedelta(days=30)).date(),
                end_date=datetime.now().date(),
                max_date_allowed=datetime.now().date(),
                display_format="YYYY-MM-DD",
                className="form-control"
            )
        ], width=4),
        dbc.Col([
            html.Label("Refresh Interval (sec)"),
            dcc.Input(id="interval-input", type="number", value=60, min=10, step=10, className="form-control")
        ], width=2)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Technical Indicators"),
            dcc.Checklist(
                id="indicator-checklist",
                options=[
                    {"label": "SMA (20)", "value": "SMA20"},
                    {"label": "EMA (20)", "value": "EMA20"},
                    {"label": "RSI (14)", "value": "RSI14"}
                ],
                value=[],
                inline=True,
                className="mb-2",
                labelStyle={"display": "inline-block", "marginRight": "20px", "color": "#444"},
                inputStyle={"marginRight": "5px"}
            )
        ], width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="historical-chart"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Refresh Data", id="refresh-button", color="primary"), width=2),
        dbc.Col(dbc.Button("Download CSV", id="download-button", color="secondary"), width=2),
        dcc.Download(id="download-dataframe-csv")
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(html.H2("Live Price", className="text-center text-success mb-2"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id="live-price", className="text-center"), width=12)
    ]),
    # Option Quotes section (contract selector)
    dbc.Row([
        dbc.Col(html.H2("Option Quotes", className="text-center text-warning mb-2"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Option Contract", className="mb-1"),
            dcc.Dropdown(
                id="option-chain-dropdown",
                options=[],
                placeholder="Select an option contract",
                className="form-control"
            )
        ], width=4)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(html.Div(id="option-data", className="text-center"), width=12)
    ], className="mb-4"),
    # Option Minute Bars section
    dbc.Row([
        dbc.Col(html.H2("Option Minute Bars", className="text-center text-primary mb-2"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="option-minute-chart"), width=12)
    ], className="mb-4"),
    # Option Time & Sales section
    dbc.Row([
        dbc.Col(html.H2("Option Time & Sales", className="text-center text-info mb-2"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Number of Option Trades", className="mb-1"),
            dcc.Input(
                id="option-trades-input",
                type="number",
                value=10,
                min=1,
                max=500,
                step=1,
                className="form-control"
            )
        ], width=2)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id="option-time-sales-table",
            columns=[
                {"name": "Time", "id": "time"},
                {"name": "Price", "id": "price"},
                {"name": "Size", "id": "size"}
            ],
            data=[],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center"}
        ), width=12)
    ], className="mb-4"),
    # NVDA Option Time & Sales section
    dbc.Row([
        dbc.Col(html.H2("NVDA Option Time & Sales", className="text-center text-warning mb-2"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Label("Option Contract (NVDA)", className="mb-1"), width=2),
        dbc.Col(dcc.Dropdown(
            id="nvda-option-chain-dropdown",
            options=[],
            placeholder="Select an NVDA option contract",
            className="form-control"
        ), width=4)
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(html.Label("Number of Trades (NVDA)", className="mb-1"), width=2),
        dbc.Col(dcc.Input(
            id="nvda-option-trades-input",
            type="number",
            value=10,
            min=1,
            max=500,
            step=1,
            className="form-control"
        ), width=2)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id="nvda-option-time-sales-table",
            columns=[
                {"name": "Time", "id": "time"},
                {"name": "Price", "id": "price"},
                {"name": "Size", "id": "size"}
            ],
            data=[],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center"}
        ), width=12)
    ], className="mb-4"),
    # Universal Option Time & Sales section
    dbc.Row([
        dbc.Col(html.H2("Universal Option Time & Sales", className="text-center text-info mb-2"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Label("Underlying", className="mb-1"), width=2),
        dbc.Col(dcc.Dropdown(
            id="ots-underlying-dropdown",
            options=[
                {"label": "Apple (AAPL)", "value": "AAPL"},
                {"label": "Microsoft (MSFT)", "value": "MSFT"},
                {"label": "Google (GOOG)", "value": "GOOG"},
                {"label": "Amazon (AMZN)", "value": "AMZN"},
                {"label": "Tesla (TSLA)", "value": "TSLA"},
                {"label": "NVIDIA (NVDA)", "value": "NVDA"}
            ],
            value="AAPL",
            clearable=False,
            className="form-control"
        ), width=4)
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(html.Label("Option Contract", className="mb-1"), width=2),
        dbc.Col(dcc.Dropdown(
            id="ots-contract-dropdown",
            options=[],
            placeholder="Select option contract",
            className="form-control"
        ), width=4)
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(html.Label("Number of Trades", className="mb-1"), width=2),
        dbc.Col(dcc.Input(
            id="ots-trades-input",
            type="number",
            value=10,
            min=1,
            max=500,
            step=1,
            className="form-control"
        ), width=2)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id="ots-time-sales-table",
            columns=[
                {"name": "Time", "id": "time"},
                {"name": "Price", "id": "price"},
                {"name": "Size", "id": "size"}
            ],
            data=[],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center"}
        ), width=12)
    ], className="mb-4"),
    # Time & Sales section
    dbc.Row([
        dbc.Col(html.H2("Time & Sales", className="text-center text-info mb-2"), width=12)
    ]),
    # Input for number of trades to fetch
    dbc.Row([
        dbc.Col([
            html.Label("Number of Trades", className="mb-1"),
            dcc.Input(
                id="trades-input",
                type="number",
                value=50,
                min=1,
                max=500,
                step=1,
                className="form-control"
            )
        ], width=2)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id="time-sales-table",
            columns=[
                {"name": "Time", "id": "time"},
                {"name": "Price", "id": "price"},
                {"name": "Size", "id": "size"}
            ],
            data=[],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center"}
        ), width=12)
    ]),
    dcc.Interval(id="interval-component", interval=60*1000, n_intervals=0)
], fluid=True)

# Update interval based on user input
@app.callback(
    Output("interval-component", "interval"),
    [Input("interval-input", "value")]
)
def update_interval_seconds(value):
    return max(value, 10) * 1000

# Update historical chart when ticker or date range changes
@app.callback(
    Output("historical-chart", "figure"),
    [Input("ticker-dropdown", "value"),
     Input("date-picker", "start_date"),
     Input("date-picker", "end_date"),
     Input("refresh-button", "n_clicks"),
     Input("indicator-checklist", "value")]
)
def update_historical_chart(ticker, start_date, end_date, n_clicks, indicators):
    df = fetch_historical(ticker.upper(), start_date, end_date)
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Candlestick(
            x=df["t"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name="OHLC"
        ))
        # Compute and plot selected indicators
        if indicators:
            # Simple Moving Average
            if "SMA20" in indicators:
                df["SMA20"] = df["c"].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=df["t"], y=df["SMA20"], mode="lines", name="SMA 20", line=dict(color="#1f77b4", width=2)))
            # Exponential Moving Average
            if "EMA20" in indicators:
                df["EMA20"] = df["c"].ewm(span=20, adjust=False).mean()
                fig.add_trace(go.Scatter(x=df["t"], y=df["EMA20"], mode="lines", name="EMA 20", line=dict(color="#ff7f0e", width=2, dash="dash")))
            # Relative Strength Index
            if "RSI14" in indicators:
                # Compute RSI (14) manually
                delta = df["c"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df["RSI14"] = 100 - (100 / (1 + rs))
                fig.add_trace(go.Scatter(x=df["t"], y=df["RSI14"], mode="lines", name="RSI 14", yaxis="y2", line=dict(color="#2ca02c", width=2)))
            # Add secondary y-axis for RSI
            fig.update_layout(
                yaxis2=dict(
                    title="RSI",
                    overlaying="y",
                    side="right",
                    showgrid=False
                )
            )
    fig.update_layout(
        title=f"Historical Data for {ticker.upper()}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        plot_bgcolor="white",
        paper_bgcolor="#f8f9fa",
        legend=dict(bgcolor="white", bordercolor="#ccc", borderwidth=1)
    )
    return fig

# Update live price at intervals
@app.callback(
    Output("live-price", "children"),
    [Input("interval-component", "n_intervals"),
     Input("ticker-dropdown", "value")]
)
def update_live_price(n_intervals, ticker):
    # Fetch the most recent stock trade using v3 endpoint
    trade = fetch_option_last_trade(ticker.upper())
    price = trade.get("p") or trade.get("price")
    ts = trade.get("sip_timestamp") or trade.get("timestamp") or trade.get("t")
    if price is not None and ts:
        dt_str = "N/A"
        try:
            # nanoseconds
            if ts > 1e16:
                dt = datetime.fromtimestamp(ts / 1e9)
            # milliseconds
            elif ts > 1e12:
                dt = datetime.fromtimestamp(ts / 1e3)
            # seconds
            elif ts > 1e9:
                dt = datetime.fromtimestamp(ts)
            else:
                dt = None
            if dt:
                dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"[LivePrice] Error parsing timestamp {ts}: {e}")
        return f"{ticker.upper()}: ${price:.2f} (as of {dt_str})"
    return f"No live data for {ticker.upper()}"

# Add CSV download callback
@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-button", "n_clicks")],
    [State("ticker-dropdown", "value"), State("date-picker", "start_date"), State("date-picker", "end_date")],
    prevent_initial_call=True
)
def download_csv(n_clicks, ticker, start_date, end_date):
    df = fetch_historical(ticker.upper(), start_date, end_date)
    return dcc.send_data_frame(df.to_csv, f"{ticker}_data_{start_date}_{end_date}.csv", index=False)

# Callback to update Time & Sales table with user-defined limit and selected date
@app.callback(
    Output("time-sales-table", "data"),
    [
        Input("interval-component", "n_intervals"),
        Input("ticker-dropdown", "value"),
        Input("date-picker", "end_date"),
        Input("trades-input", "value")
    ]
)
def update_time_sales(n_intervals, ticker, end_date, limit):
    df = fetch_time_sales(ticker.upper(), date=end_date, limit=limit)
    if df.empty:
        # fallback: 15-min delayed 1-min bars over last 30 days
        for days_back in range(0, 30):
            try_date = (datetime.fromisoformat(end_date) - timedelta(days=days_back)).strftime("%Y-%m-%d")
            bars = fetch_stock_minute_bars(ticker.upper(), try_date, try_date)
            if not bars.empty:
                df2 = pd.DataFrame({"time": bars["t"], "price": bars["c"], "size": bars["v"]})
                df2 = df2.sort_values("time", ascending=False).head(limit)
                df2["time"] = df2["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
                return df2.to_dict("records")
        return []
    # real-time data available
    df = df.sort_values("time", ascending=False).head(limit)
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return df.to_dict("records")

# Callback to update Option Quotes/Snapshot in UI
@app.callback(
    Output("option-data", "children"),
    [
        Input("interval-component", "n_intervals"),
        Input("ticker-dropdown", "value"),
        Input("option-chain-dropdown", "value")
    ]
)
def update_option_data(n_intervals, underlying, contract):
    if not underlying or not contract:
        return "Select an underlying and contract above"
    # Fetch latest option trade and quote
    lt = fetch_option_last_trade(contract)
    lq = fetch_option_last_quote(contract)
    children = []
    # Last trade
    price = lt.get("p") or lt.get("price")
    size = lt.get("s") or lt.get("size")
    ts = lt.get("sip_timestamp") or lt.get("timestamp") or lt.get("t")
    if price is not None:
        dt_str = "N/A"
        if ts:
            try:
                if ts > 1e16:
                    dt = datetime.fromtimestamp(ts / 1e9)
                elif ts > 1e12:
                    dt = datetime.fromtimestamp(ts / 1e3)
                elif ts > 1e9:
                    dt = datetime.fromtimestamp(ts)
                else:
                    dt = None
                if dt:
                    dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                print(f"[OptionData] Error parsing trade timestamp {ts}: {e}")
        children.append(html.P(f"Trade → Price: ${price:.2f}, Size: {size}, Time: {dt_str}"))
    else:
        children.append(html.P("No trade data"))
    # Last quote
    bid = lq.get("bidprice") or lq.get("bp") or lq.get("bid")
    bidsize = lq.get("bidsize") or lq.get("bidsz")
    ask = lq.get("askprice") or lq.get("ap") or lq.get("ask")
    asksize = lq.get("asksize") or lq.get("asksz")
    ts2 = lq.get("sip_timestamp") or lq.get("timestamp")
    if bid is not None and ask is not None:
        dt2_str = "N/A"
        if ts2:
            try:
                if ts2 > 1e16:
                    dt2 = datetime.fromtimestamp(ts2 / 1e9)
                elif ts2 > 1e12:
                    dt2 = datetime.fromtimestamp(ts2 / 1e3)
                elif ts2 > 1e9:
                    dt2 = datetime.fromtimestamp(ts2)
                else:
                    dt2 = None
                if dt2:
                    dt2_str = dt2.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                print(f"[OptionData] Error parsing quote timestamp {ts2}: {e}")
        children.append(html.P(f"Quote → Bid: ${bid:.2f} ({bidsize}), Ask: ${ask:.2f} ({asksize}), Time: {dt2_str}"))
    else:
        children.append(html.P("No quote data"))
    return children

# Callback to update Option Minute Bars chart
@app.callback(
    Output("option-minute-chart", "figure"),
    [
        Input("option-chain-dropdown", "value"),
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("refresh-button", "n_clicks")
    ]
)
def update_option_minute_chart(option_ticker, start_date, end_date, n_clicks):
    fig = go.Figure()
    if option_ticker and start_date and end_date:
        # Attempt to fetch minute bars for the selected range
        df = fetch_option_minute_bars(option_ticker, start_date, end_date)
        # If no bars found, extend lookup backward up to 30 days
        if df.empty:
            for days_back in range(1, 31):
                alt_start = (datetime.fromisoformat(end_date) - timedelta(days=days_back)).strftime("%Y-%m-%d")
                df_alt = fetch_option_minute_bars(option_ticker, alt_start, end_date)
                if not df_alt.empty:
                    df = df_alt
                    break
        # Plot if data exists, else show annotation
        if not df.empty:
            fig.add_trace(go.Candlestick(
                x=df["t"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name=option_ticker
            ))
            fig.update_layout(
                title=f"Minute Bars for {option_ticker}",
                xaxis_title="Time",
                yaxis_title="Price",
                plot_bgcolor="white",
                paper_bgcolor="#f8f9fa"
            )
        else:
            fig.add_annotation(
                text="No option bars available for selected period or last 30 days",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
    return fig

# Callback to update Option Time & Sales table with user-defined limit and selected contract
@app.callback(
    Output("option-time-sales-table", "data"),
    [
        Input("interval-component", "n_intervals"),
        Input("option-chain-dropdown", "value"),
        Input("option-trades-input", "value")
    ]
)
def update_option_time_sales(n_intervals, contract, limit):
    # Don't run if no contract selected
    if not contract:
        return []
    df = fetch_option_trades(contract, limit=limit)
    if df.empty:
        # fallback: 15-min delayed minute bars over last 30 days
        for days_back in range(0, 30):
            try_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            bars = fetch_option_minute_bars(contract, try_date, try_date)
            if not bars.empty:
                df2 = pd.DataFrame({"time": bars["t"], "price": bars["c"], "size": bars.get("v", pd.Series())})
                df2 = df2.sort_values("time", ascending=False).head(limit)
                df2["time"] = df2["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
                return df2.to_dict("records")
        return []
    df = df.sort_values("time", ascending=False).head(limit)
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return df.to_dict("records")

# Populate option-chain-dropdown based on underlying
@app.callback(
    [Output("option-chain-dropdown", "options"), Output("option-chain-dropdown", "value")],
    [Input("ticker-dropdown", "value")]
)
def update_option_chain(underlying):
    if not underlying:
        return [], None
    contracts = fetch_option_contracts(underlying)
    opts = [{"label": c["ticker"], "value": c["ticker"]} for c in contracts]
    default = opts[0]["value"] if opts else None
    return opts, default

# Callback to update NVDA option chain (fixed underlying NVDA)
@app.callback(
    [Output("nvda-option-chain-dropdown", "options"), Output("nvda-option-chain-dropdown", "value")],
    [Input("interval-component", "n_intervals")]
)
def update_nvda_option_chain(n_intervals):
    contracts = fetch_option_contracts("NVDA")
    opts = [{"label": c["ticker"], "value": c["ticker"]} for c in contracts]
    default = opts[0]["value"] if opts else None
    return opts, default

# Callback to update NVDA Option Time & Sales table
@app.callback(
    Output("nvda-option-time-sales-table", "data"),
    [
        Input("interval-component", "n_intervals"),
        Input("nvda-option-chain-dropdown", "value"),
        Input("nvda-option-trades-input", "value")
    ]
)
def update_nvda_option_time_sales(n_intervals, contract, limit):
    # Don't run if no NVDA contract selected
    if not contract:
        return []
    df = fetch_option_trades(contract, limit=limit)
    if df.empty:
        # fallback: 15-min delayed minute bars over last 30 days
        for days_back in range(0, 30):
            try_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            bars = fetch_option_minute_bars(contract, try_date, try_date)
            if not bars.empty:
                df2 = pd.DataFrame({"time": bars["t"], "price": bars["c"], "size": bars.get("v", pd.Series())})
                df2 = df2.sort_values("time", ascending=False).head(limit)
                df2["time"] = df2["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
                return df2.to_dict("records")
        return []
    df = df.sort_values("time", ascending=False).head(limit)
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return df.to_dict("records")

# Callback to populate Universal Option contract dropdown
@app.callback(
    [Output("ots-contract-dropdown", "options"), Output("ots-contract-dropdown", "value")],
    [Input("ots-underlying-dropdown", "value")]
)
def update_ots_contract_dropdown(underlying):
    if not underlying:
        return [], None
    contracts = fetch_option_contracts(underlying)
    opts = [{"label": c["ticker"], "value": c["ticker"]} for c in contracts]
    default = opts[0]["value"] if opts else None
    return opts, default

# Callback to update Universal Option Time & Sales table
@app.callback(
    Output("ots-time-sales-table", "data"),
    [
        Input("interval-component", "n_intervals"),
        Input("ots-contract-dropdown", "value"),
        Input("ots-trades-input", "value")
    ]
)
def update_ots_time_sales(n_intervals, contract, limit):
    # Don't run if no contract selected
    if not contract:
        return []
    df = fetch_option_trades(contract, limit=limit)
    if df.empty:
        # fallback: 15-min delayed minute bars over last 30 days
        for days_back in range(0, 30):
            try_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            bars = fetch_option_minute_bars(contract, try_date, try_date)
            if not bars.empty:
                df2 = pd.DataFrame({"time": bars["t"], "price": bars["c"], "size": bars.get("v", pd.Series())})
                df2 = df2.sort_values("time", ascending=False).head(limit)
                df2["time"] = df2["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
                return df2.to_dict("records")
        return []
    df = df.sort_values("time", ascending=False).head(limit)
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return df.to_dict("records")

if __name__ == "__main__":
    # Launch server via app.run (replacing run_server) on localhost, disable hot reload
    app.run(debug=True, host="127.0.0.1", port=int(os.environ.get("PORT", 8050)), dev_tools_hot_reload=False) 