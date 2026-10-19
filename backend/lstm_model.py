import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class StockLSTMModel:
    def __init__(self, input_shape=(60, 1)):
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        print("Model compiled successfully!")

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the LSTM with early stopping"""
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True,
            callbacks=[early_stop],
            verbose=1
        )
        return history

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, mae

    def save(self, filepath):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")