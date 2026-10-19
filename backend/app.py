import traceback
import warnings

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import BATCH_SIZE, EPOCHS, LOOKBACK, YEARS_OF_DATA
from data_processor import StockDataProcessor
from lstm_model import StockLSTMModel

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Stock Price Predictor API is running!'})


@app.route('/api/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True) or {}
    symbol = payload.get('symbol', 'AAPL').upper()
    print(f"\n{'=' * 50}\nPredicting for {symbol}\n{'=' * 50}")

    try:
        processor = StockDataProcessor(symbol, years=YEARS_OF_DATA)
        X_train, y_train, X_test, y_test = processor.prepare_data(lookback=LOOKBACK)

        model = StockLSTMModel(input_shape=(LOOKBACK, 1))
        model.train(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

        predicted = processor.inverse_transform(model.predict(X_test)).flatten()
        actual = processor.inverse_transform(y_test.reshape(-1, 1)).flatten()

        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(mean_absolute_error(actual, predicted))
        # Naive baseline: tomorrow's price equals today's price
        naive_rmse = float(np.sqrt(mean_squared_error(actual[1:], actual[:-1])))

        print(f"RMSE: ${rmse:.2f} | Naive RMSE: ${naive_rmse:.2f} | MAE: ${mae:.2f}")

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'last_close': processor.get_latest_price(),
            'last_date': processor.data.index[-1].strftime('%Y-%m-%d'),
            'dates': processor.get_test_dates(),
            'actual': actual.tolist(),
            'predicted': predicted.tolist(),
            'rmse': rmse,
            'mae': mae,
            'naive_rmse': naive_rmse,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)