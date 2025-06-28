class GAFCNNCandlestickClassifier:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.pattern_names = [
            'Other', 'Morning Star', 'Bullish Engulfing', 'Hammer', 'Inverted Hammer',
            'Evening Star', 'Bearish Engulfing', 'Shooting Star', 'Hanging Man'
        ]
        self.training_history = None
        self.test_accuracy = None

    def calculate_culr_features(self, df):
        result_df = df.copy()

        result_df['upper_shadow'] = result_df['high'] - result_df[['open', 'close']].max(axis=1)
        result_df['lower_shadow'] = result_df[['open', 'close']].min(axis=1) - result_df['low']
        result_df['real_body'] = result_df['close'] - result_df['open']

        return result_df[['close', 'upper_shadow', 'lower_shadow', 'real_body']]

    def is_doji(self, open_price, close_price, high_price, low_price, threshold=0.1):
        body_size = abs(close_price - open_price)
        range_size = high_price - low_price
        return body_size <= (range_size * threshold) if range_size > 0 else True

    def is_long_body(self, open_price, close_price, avg_body_size):
        body_size = abs(close_price - open_price)
        return body_size > (avg_body_size * 1.2)

    def is_small_body(self, open_price, close_price, avg_body_size):
        body_size = abs(close_price - open_price)
        return body_size < (avg_body_size * 0.3)

    def detect_hammer(self, open_price, high_price, low_price, close_price):
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        total_range = high_price - low_price

        if total_range == 0 or body_size == 0:
            return False

        return (lower_shadow >= 1.5 * body_size and
                upper_shadow <= body_size * 0.2 and
                body_size < total_range * 0.4)

    def detect_inverted_hammer(self, open_price, high_price, low_price, close_price):
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        total_range = high_price - low_price

        if total_range == 0 or body_size == 0:
            return False

        return (upper_shadow >= 1.5 * body_size and
                lower_shadow <= body_size * 0.2 and
                body_size < total_range * 0.4)

    def detect_shooting_star(self, open_price, high_price, low_price, close_price, prev_trend='up'):
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        total_range = high_price - low_price

        if total_range == 0 or body_size == 0:
            return False

        return (upper_shadow >= 1.5 * body_size and
                lower_shadow <= body_size * 0.2 and
                body_size < total_range * 0.4 and
                prev_trend == 'up')

    def detect_hanging_man(self, open_price, high_price, low_price, close_price, prev_trend='up'):
        is_hammer_shape = self.detect_hammer(open_price, high_price, low_price, close_price)
        return is_hammer_shape and prev_trend == 'up'

    def detect_bullish_engulfing(self, prev_open, prev_high, prev_low, prev_close,
                                curr_open, curr_high, curr_low, curr_close):
        prev_bearish = prev_close < prev_open
        curr_bullish = curr_close > curr_open

        curr_engulfs_prev = (curr_open <= prev_close + 0.5 * abs(prev_close - prev_open) and
                           curr_close > prev_open)

        return prev_bearish and curr_bullish and curr_engulfs_prev

    def detect_bearish_engulfing(self, prev_open, prev_high, prev_low, prev_close,
                                curr_open, curr_high, curr_low, curr_close):
        prev_bullish = prev_close > prev_open
        curr_bearish = curr_close < curr_open

        curr_engulfs_prev = (curr_open >= prev_close - 0.5 * abs(prev_close - prev_open) and
                           curr_close < prev_open)

        return prev_bullish and curr_bearish and curr_engulfs_prev

    def detect_morning_star(self, first_open, first_high, first_low, first_close,
                           second_open, second_high, second_low, second_close,
                           third_open, third_high, third_low, third_close):
        avg_body = np.mean([
            abs(first_close - first_open),
            abs(second_close - second_open),
            abs(third_close - third_open)
        ])

        first_bearish = first_close < first_open
        first_long = abs(first_close - first_open) > avg_body * 0.8

        second_small = abs(second_close - second_open) < avg_body * 0.5

        third_bullish = third_close > third_open
        third_long = abs(third_close - third_open) > avg_body * 0.8
        third_closes_above_midpoint = third_close > (first_open + first_close) / 2

        return (first_bearish and first_long and second_small and
                third_bullish and third_long and third_closes_above_midpoint)

    def detect_evening_star(self, first_open, first_high, first_low, first_close,
                           second_open, second_high, second_low, second_close,
                           third_open, third_high, third_low, third_close):
        avg_body = np.mean([
            abs(first_close - first_open),
            abs(second_close - second_open),
            abs(third_close - third_open)
        ])

        first_bullish = first_close > first_open
        first_long = abs(first_close - first_open) > avg_body * 0.8

        second_small = abs(second_close - second_open) < avg_body * 0.5

        third_bearish = third_close < third_open
        third_long = abs(third_close - third_open) > avg_body * 0.8
        third_closes_below_midpoint = third_close < (first_open + first_close) / 2

        return (first_bullish and first_long and second_small and
                third_bearish and third_long and third_closes_below_midpoint)

    def determine_trend(self, closes, period=5):
        if len(closes) < period:
            return 'neutral'

        recent_closes = closes[-period:]
        slope = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]

        if slope > np.std(recent_closes) * 0.1:
            return 'up'
        elif slope < -np.std(recent_closes) * 0.1:
            return 'down'
        else:
            return 'neutral'

    def identify_candlestick_patterns(self, df):
        patterns = []

        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        body_sizes = np.abs(closes - opens)
        avg_body_size = np.mean(body_sizes[body_sizes > 0])

        for i in range(len(df) - self.window_size + 1):
            window_start = i
            window_end = i + self.window_size

            window_opens = opens[window_start:window_end]
            window_highs = highs[window_start:window_end]
            window_lows = lows[window_start:window_end]
            window_closes = closes[window_start:window_end]

            pattern_found = 0

            if self.window_size >= 3:
                if self.detect_morning_star(
                    window_opens[-3], window_highs[-3], window_lows[-3], window_closes[-3],
                    window_opens[-2], window_highs[-2], window_lows[-2], window_closes[-2],
                    window_opens[-1], window_highs[-1], window_lows[-1], window_closes[-1]
                ):
                    pattern_found = 1
                elif self.detect_evening_star(
                    window_opens[-3], window_highs[-3], window_lows[-3], window_closes[-3],
                    window_opens[-2], window_highs[-2], window_lows[-2], window_closes[-2],
                    window_opens[-1], window_highs[-1], window_lows[-1], window_closes[-1]
                ):
                    pattern_found = 5

            if self.window_size >= 2 and pattern_found == 0:
                if self.detect_bullish_engulfing(
                    window_opens[-2], window_highs[-2], window_lows[-2], window_closes[-2],
                    window_opens[-1], window_highs[-1], window_lows[-1], window_closes[-1]
                ):
                    pattern_found = 2
                elif self.detect_bearish_engulfing(
                    window_opens[-2], window_highs[-2], window_lows[-2], window_closes[-2],
                    window_opens[-1], window_highs[-1], window_lows[-1], window_closes[-1]
                ):
                    pattern_found = 6

            if pattern_found == 0:
                trend = self.determine_trend(window_closes[:-1])

                if self.detect_hammer(window_opens[-1], window_highs[-1],
                                    window_lows[-1], window_closes[-1]):
                    pattern_found = 3
                elif self.detect_inverted_hammer(window_opens[-1], window_highs[-1],
                                               window_lows[-1], window_closes[-1]):
                    pattern_found = 4
                elif self.detect_shooting_star(window_opens[-1], window_highs[-1],
                                             window_lows[-1], window_closes[-1], trend):
                    pattern_found = 7
                elif self.detect_hanging_man(window_opens[-1], window_highs[-1],
                                           window_lows[-1], window_closes[-1], trend):
                    pattern_found = 8

            patterns.append(pattern_found)

        return patterns

    def gramian_angular_field(self, ts):
        min_ts = np.min(ts)
        max_ts = np.max(ts)

        if max_ts == min_ts:
            scaled_ts = np.full_like(ts, 0.5)
        else:
            scaled_ts = (ts - min_ts) / (max_ts - min_ts)

        scaled_ts = np.clip(scaled_ts, 0, 1)
        phi = np.arccos(scaled_ts)
        gaf = np.cos(phi[:, None] + phi[None, :])

        return gaf

    def prepare_gaf_cnn_data(self, df):
        data = []

        for i in range(len(df) - self.window_size + 1):
            window = df.iloc[i:i+self.window_size].values

            gaf_matrices = []
            for feature_idx in range(window.shape[1]):
                ts = window[:, feature_idx]
                gaf = self.gramian_angular_field(ts)
                gaf_matrices.append(gaf)

            gaf_stack = np.stack(gaf_matrices, axis=2)
            data.append(gaf_stack)

        return np.array(data)

    def build_model(self):
        input_shape = (self.window_size, self.window_size, 4)

        model = Sequential([
            Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            Conv2D(16, kernel_size=(3, 3), activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(9, activation='softmax')
        ])

        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model
        return model

    def train_and_evaluate(self, df, epochs=300, batch_size=64, test_size=0.2):
        print("Calculating CULR features...")
        culr_df = self.calculate_culr_features(df)

        print("Preparing GAF-CNN data...")
        X = self.prepare_gaf_cnn_data(culr_df)

        print("Identifying candlestick patterns...")
        pattern_labels = self.identify_candlestick_patterns(df)
        y = to_categorical(pattern_labels, num_classes=9)

        unique, counts = np.unique(pattern_labels, return_counts=True)
        print("\nPattern distribution:")
        total_samples = len(pattern_labels)
        for pattern_id, count in zip(unique, counts):
            percentage = count/total_samples*100
            print(f"{self.pattern_names[pattern_id]}: {count} samples ({percentage:.1f}%)")

        print(f"\nGenerated {X.shape[0]} samples with shape {X.shape[1:]}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=np.argmax(y, axis=1)
        )

        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")

        print("Building model...")
        self.build_model()
        self.model.summary()

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )

        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )

        self.training_history = history

        print("\n" + "="*60)
        print("FINAL MODEL EVALUATION")
        print("="*60)

        test_predictions = self.model.predict(X_test)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        test_true_classes = np.argmax(y_test, axis=1)

        self.test_accuracy = accuracy_score(test_true_classes, test_pred_classes)

        print(f"\nTest Accuracy: {self.test_accuracy:.4f} ({self.test_accuracy*100:.2f}%)")
        print(f"Paper's Reported Accuracy: 90.7%")


        cm = confusion_matrix(test_true_classes, test_pred_classes)

        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.pattern_names,
                   yticklabels=self.pattern_names)
        plt.title(f'Confusion Matrix - Test Accuracy: {self.test_accuracy:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        return history

    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.axhline(y=0.907, color='red', linestyle='--', label='Paper Accuracy (90.7%)', linewidth=2)
        ax1.set_title(f'Model Accuracy\nFinal Test Accuracy: {self.test_accuracy:.4f}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def main():
    print("GAF-CNN CANDLESTICK PATTERN IDENTIFICATION")
    print("="*60)

    print("\nLoading RELIANCE.csv...")
    try:
        df = pd.read_csv('RELIANCE.csv')
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"First few rows:")

        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Required columns {required_columns} not found in data")

        classifier = GAFCNNCandlestickClassifier(window_size=10)

        print("\n" + "="*60)
        print("STARTING TRAINING WITH REAL PATTERN RECOGNITION")
        print("="*60)

        history = classifier.train_and_evaluate(
            df=df,
            epochs=50,
            batch_size=64,
            test_size=0.2
        )

        classifier.plot_training_history(history)

        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Dataset: RELIANCE.csv ({len(df):,} candlesticks)")
        print(f"Window Size: {classifier.window_size}")
        print(f"Architecture: GAF-CNN (2 Conv layers, 1 FC layer)")
        print(f"Final Test Accuracy: {classifier.test_accuracy:.4f} ({classifier.test_accuracy*100:.2f}%)")
        print(f"Paper Accuracy: 90.7%")

        print("\n" + "="*60)

    except FileNotFoundError:
        print("Error: RELIANCE.csv file not found.")
        print("Please ensure the file is in the current directory with lowercase column names:")
        print("Required columns: ['open', 'high', 'low', 'close']")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
