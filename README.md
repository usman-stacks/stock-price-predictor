**Stock Price Predictor (LSTM)**

I built this to learn how a neural network handles time-series data, and to connect that model to a small web app. It downloads five years of daily prices for a stock, trains an LSTM on the earlier part of that history, and checks its predictions against the part it hasn't seen.

This is a learning project, not financial advice.

**What I found**

I tested it on [SYMBOL], using [START] to [END] as the test period.

- Model error (RMSE): $[value]
- Naive error (tomorrow = today): $[value]
- Average error (MAE): $[value]

[Write one or two sentences about what you saw. If the model did worse than the naive guess, say that plainly. Beating "tomorrow equals today" is hard for price-only models, and an honest result is still a result.]

**How it works**

The backend downloads the closing prices and scales them to a 0 to 1 range. I fit the scaler on the training period only, so the test data doesn't leak into training. Each training example is 60 days of prices, and the target is the next day's price. The model is a stacked LSTM with dropout, and early stopping ends training once validation loss stops improving. The final 20% of the data is kept for testing, and the predictions are sent to the frontend, which draws them against the real prices.

**Built with**

Python 3.12, TensorFlow/Keras, scikit-learn, pandas, and NumPy for the model. yfinance for market data. Flask and Flask-CORS for the API. Plotly.js for the chart.

**Getting it running**

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/usman-stacks/stock-price-predictor.git
cd stock-price-predictor
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Start the backend in one terminal:

```bash
cd backend
python3 app.py
```

Start the frontend in a second terminal:

```bash
cd frontend
python3 -m http.server 8000
```

Then open http://localhost:8000, pick a stock, and click Predict Price. The first run takes a minute or two, since the model trains each time you ask for a prediction.

**Project layout**

```
stock-price-predictor/
├── backend/
│   ├── app.py              # Flask API
│   ├── config.py           # Settings: symbols, lookback, epochs
│   ├── data_processor.py   # Download, scale, and window the data
│   └── lstm_model.py       # LSTM architecture and training
├── frontend/
│   └── index.html          # Dashboard
├── requirements.txt
└── README.md
```

**Things to know**

The model only sees past closing prices. It has no idea about news, earnings, or anything else that moves a stock. Prices behave close to a random walk, so predicting tomorrow from history is hard. Results also change between runs, because the model retrains from scratch every time. Yahoo Finance sometimes rate-limits requests too. If a stock returns no data, wait a few minutes and try again.

**What I'd try next**

- Save trained models so results are consistent between runs
- Add features like volume or moving averages, and see if they beat the baseline
- Test across several different time periods, not just one