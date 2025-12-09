import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_close(symbol, date):
    start = date
    end = date + timedelta(days=1)

    data = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        return None

    # Use 'Adj Close' if available, otherwise 'Close'
    col = "Adj Close" if "Adj Close" in data.columns else "Close"
    value = data[col].iloc[0]

    if hasattr(value, "item"):
        return value.item()
    return float(value)

def get_next_trading_day(date):
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    return next_day

def label_row(row):
    post_dt = datetime.utcfromtimestamp(int(row["created_utc"]))
    post_date = post_dt.date()

    # If Saturday (5) or Sunday (6), default to next Monday
    if post_date.weekday() >= 5:
        baseline_date = post_date + timedelta(days=(8 - post_date.weekday()))
        next_trading_date = baseline_date + timedelta(days=1)  # Tuesday
    else:
        # Weekday post: use post date as baseline, next trading day after
        baseline_date = post_date
        next_trading_date = get_next_trading_day(post_date)

    symbol = row["stock_symbol"].upper()

    baseline_close = get_close(symbol, baseline_date)
    next_close = get_close(symbol, next_trading_date)

    if baseline_close is None or next_close is None:
        return -1

    return 1 if next_close > baseline_close else 0

def main():
    df = pd.read_csv("posts_with_symbols.csv")
    df = df.head(2000)
    df["next_day_movement"] = df.apply(label_row, axis=1)
    df.to_csv("labeled.csv", index=False)

if __name__ == "__main__":
    main()
