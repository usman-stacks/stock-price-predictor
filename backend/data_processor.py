from datetime import date, timedelta

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class StockDataProcessor:
    def __init__(self, symbol, years=5):
        self.symbol = symbol.upper()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.split = None

        end = date.today()
        start = end - timedelta(days=365 * years)
        self.data = yf.download(
            self.symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
        )
        if self.data.empty:
            raise ValueError(
                f"No data returned for {self.symbol}. Yahoo Finance may be "
                "rate-limiting requests; wait a few minutes and try again."
            )

    def _close_prices(self):
        return self.data['Close'].to_numpy().reshape(-1, 1)

    def prepare_data(self, lookback=60, train_ratio=0.8):
        """Build sequences. The scaler is fitted on training prices only."""
        close = self._close_prices()
        self.split = int(len(close) * train_ratio)

        self.scaler.fit(close[:self.split])
        scaled = self.scaler.transform(close)

        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i - lookback:i, 0])
            y.append(scaled[i, 0])

        X = np.array(X, dtype=np.float32).reshape(-1, lookback, 1)
        y = np.array(y, dtype=np.float32)

        # Sample k predicts day k + lookback. Training samples are those
        # whose target day falls before the split.
        n_train = self.split - lookback
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    def inverse_transform(self, scaled_values):
        return self.scaler.inverse_transform(scaled_values)

    def get_latest_price(self):
        return float(self._close_prices()[-1, 0])

    def get_test_dates(self):
        return [d.strftime('%Y-%m-%d') for d in self.data.index[self.split:]]