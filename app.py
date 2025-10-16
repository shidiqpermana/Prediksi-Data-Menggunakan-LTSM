"""
Bike Sharing Demand - Flask Web Application
Aplikasi web untuk menampilkan data historis dan prediksi bike sharing demand
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
import io
import base64
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for model and scalers
model = None
feature_scaler = None
target_scaler = None
feature_names = None

def load_model_and_scalers():
    """Load the trained model and scalers"""
    global model, feature_scaler, target_scaler, feature_names
    
    try:
        import os
        
        # Check if model files exist
        model_path = 'models/lstm_model.h5'
        feature_scaler_path = 'models/feature_scaler.pkl'
        target_scaler_path = 'models/target_scaler.pkl'
        feature_names_path = 'models/feature_names.npy'
        
        for path in [model_path, feature_scaler_path, target_scaler_path, feature_names_path]:
            if not os.path.exists(path):
                print(f"Error: Model file not found: {path}")
                return False
        
        # Load model with custom objects to handle compatibility issues
        from tensorflow.keras.utils import get_custom_objects
        import tensorflow.keras.metrics as metrics
        
        # Define custom objects for compatibility
        custom_objects = {
            'mse': 'mse',
            'mae': 'mae',
            'mean_squared_error': 'mse',
            'mean_absolute_error': 'mae'
        }
        
        print("Loading LSTM model...")
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        
        # Recompile the model with current TensorFlow version
        from tensorflow.keras.optimizers import Adam
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("Loading scalers...")
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        feature_names = np.load(feature_names_path)
        
        print("Model and scalers loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Feature names: {feature_names}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_time_features(datetime_obj):
    """Create time features for a given datetime"""
    hour = datetime_obj.hour
    dayofweek = datetime_obj.weekday()
    month = datetime_obj.month
    dayofyear = datetime_obj.timetuple().tm_yday
    
    is_weekend = 1 if dayofweek >= 5 else 0
    is_rush_hour = 1 if ((hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19)) else 0
    is_peak_hour = 1 if ((hour >= 8 and hour <= 10) or (hour >= 18 and hour <= 20)) else 0
    
    return {
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'dayofyear': dayofyear,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour,
        'is_peak_hour': is_peak_hour
    }

def predict_demand(weather_data, datetime_obj):
    """Predict bike demand for given conditions"""
    global model, feature_scaler, target_scaler, feature_names
    
    try:
        # Check if model is loaded
        if model is None or feature_scaler is None or target_scaler is None or feature_names is None:
            raise ValueError("Model or scalers not loaded properly")
        
        # Create time features
        time_features = create_time_features(datetime_obj)
        
        # Debug logging
        print(f"Predicting for datetime: {datetime_obj}")
        print(f"Time features: {time_features}")
        print(f"Weather data: {weather_data}")
        
        # Combine weather and time features
        features = {
            'season': weather_data.get('season', 1),
            'holiday': weather_data.get('holiday', 0),
            'workingday': weather_data.get('workingday', 1),
            'weather': weather_data.get('weather', 1),
            'temp': weather_data.get('temp', 20),
            'atemp': weather_data.get('atemp', 20),
            'humidity': weather_data.get('humidity', 50),
            'windspeed': weather_data.get('windspeed', 10),
            **time_features
        }
        
        # Convert to array in correct order
        feature_array = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        # Scale features
        scaled_features = feature_scaler.transform(feature_array)
        
        # For single prediction, we need to create a sequence
        # Create a more realistic sequence by varying some features slightly
        sequence_length = 24
        sequence = []
        
        # Create a sequence with slight variations to make predictions more responsive
        for i in range(sequence_length):
            # Add small variations to make the sequence more realistic
            variation_factor = 1 + (i - sequence_length//2) * 0.02  # Small variation
            
            varied_features = scaled_features.copy()
            # Vary temperature slightly
            temp_idx = list(feature_names).index('temp') if 'temp' in feature_names else -1
            if temp_idx >= 0:
                varied_features[0, temp_idx] *= variation_factor
            
            # Vary hour for more realistic sequence
            hour_idx = list(feature_names).index('hour') if 'hour' in feature_names else -1
            if hour_idx >= 0:
                # Create a realistic hour progression
                base_hour = time_features['hour']
                sequence_hour = (base_hour - sequence_length//2 + i) % 24
                # Recreate time features for this hour
                temp_datetime = datetime_obj.replace(hour=sequence_hour)
                temp_time_features = create_time_features(temp_datetime)
                # Update hour-related features
                for key, value in temp_time_features.items():
                    if key in feature_names:
                        idx = list(feature_names).index(key)
                        varied_features[0, idx] = value
            
            sequence.append(varied_features[0])
        
        sequence = np.array(sequence).reshape(1, sequence_length, -1)
        
        # Make prediction
        prediction = model.predict(sequence, verbose=0)
        
        # Inverse transform
        prediction_original = target_scaler.inverse_transform(prediction.reshape(-1, 1))
        
        result = max(0, int(prediction_original[0][0]))
        print(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        print(f"Error in predict_demand: {e}")
        import traceback
        traceback.print_exc()
        # Return a default prediction if there's an error
        return 100

def create_plot_base64():
    """Create a plot and return as base64 string"""
    # Load historical data
    train_df = pd.read_csv('train.csv')
    train_df['datetime'] = pd.to_datetime(train_df['datetime'])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['datetime'], train_df['count'], linewidth=0.8, alpha=0.7)
    plt.title('Historical Bike Sharing Demand', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Number of Bikes Rented')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

def create_pattern_plots():
    """Create pattern analysis plots"""
    train_df = pd.read_csv('train.csv')
    train_df['datetime'] = pd.to_datetime(train_df['datetime'])
    train_df['hour'] = train_df['datetime'].dt.hour
    train_df['dayofweek'] = train_df['datetime'].dt.dayofweek
    train_df['month'] = train_df['datetime'].dt.month
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Hourly pattern
    hourly_pattern = train_df.groupby('hour')['count'].mean()
    axes[0, 0].plot(hourly_pattern.index, hourly_pattern.values, marker='o', linewidth=2)
    axes[0, 0].set_title('Average Hourly Demand')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Average Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Daily pattern
    daily_pattern = train_df.groupby('dayofweek')['count'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[0, 1].plot(days, daily_pattern.values, marker='o', linewidth=2)
    axes[0, 1].set_title('Average Daily Demand')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Average Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Monthly pattern
    monthly_pattern = train_df.groupby('month')['count'].mean()
    axes[1, 0].plot(monthly_pattern.index, monthly_pattern.values, marker='o', linewidth=2)
    axes[1, 0].set_title('Average Monthly Demand')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Weather impact
    weather_pattern = train_df.groupby('weather')['count'].mean()
    axes[1, 1].bar(weather_pattern.index, weather_pattern.values)
    axes[1, 1].set_title('Demand by Weather Condition')
    axes[1, 1].set_xlabel('Weather (1=Clear, 2=Mist, 3=Light Rain, 4=Heavy Rain)')
    axes[1, 1].set_ylabel('Average Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    """Main page"""
    # Load model if not already loaded
    if model is None:
        if not load_model_and_scalers():
            return render_template('error.html', message="Model could not be loaded. Please ensure the model files exist.")
    
    # Create plots
    historical_plot = create_plot_base64()
    pattern_plots = create_pattern_plots()
    
    return render_template('index.html', 
                         historical_plot=historical_plot,
                         pattern_plots=pattern_plots)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Parse datetime
        datetime_str = data.get('datetime')
        if not datetime_str:
            return jsonify({'error': 'Datetime is required'}), 400
        
        # Handle both ISO format (with T) and standard format (with space)
        try:
            if 'T' in datetime_str:
                datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S')
            else:
                datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            return jsonify({'error': f'Invalid datetime format: {datetime_str}. Expected format: YYYY-MM-DD HH:MM:SS or YYYY-MM-DDTHH:MM:SS'}), 400
        
        # Get weather data
        weather_data = {
            'season': int(data.get('season', 1)),
            'holiday': int(data.get('holiday', 0)),
            'workingday': int(data.get('workingday', 1)),
            'weather': int(data.get('weather', 1)),
            'temp': float(data.get('temp', 20)),
            'atemp': float(data.get('atemp', 20)),
            'humidity': float(data.get('humidity', 50)),
            'windspeed': float(data.get('windspeed', 10))
        }
        
        # Make prediction
        prediction = predict_demand(weather_data, datetime_obj)
        
        return jsonify({
            'prediction': prediction,
            'datetime': datetime_str,
            'weather_conditions': weather_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/forecast', methods=['POST'])
def forecast():
    """Generate forecast for next 24 hours"""
    try:
        data = request.get_json()
        
        # Get starting datetime
        start_datetime_str = data.get('start_datetime')
        if not start_datetime_str:
            start_datetime = datetime.now()
        else:
            # Handle both ISO format (with T) and standard format (with space)
            try:
                if 'T' in start_datetime_str:
                    start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%dT%H:%M:%S')
                else:
                    start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                return jsonify({'error': f'Invalid datetime format: {start_datetime_str}. Expected format: YYYY-MM-DD HH:MM:SS or YYYY-MM-DDTHH:MM:SS'}), 400
        
        # Get weather data
        weather_data = {
            'season': int(data.get('season', 1)),
            'holiday': int(data.get('holiday', 0)),
            'workingday': int(data.get('workingday', 1)),
            'weather': int(data.get('weather', 1)),
            'temp': float(data.get('temp', 20)),
            'atemp': float(data.get('atemp', 20)),
            'humidity': float(data.get('humidity', 50)),
            'windspeed': float(data.get('windspeed', 10))
        }
        
        # Generate 24-hour forecast
        forecast_data = []
        for i in range(24):
            forecast_datetime = start_datetime + timedelta(hours=i)
            prediction = predict_demand(weather_data, forecast_datetime)
            
            forecast_data.append({
                'datetime': forecast_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'prediction': prediction,
                'hour': forecast_datetime.hour,
                'day': forecast_datetime.strftime('%A')
            })
        
        return jsonify({
            'forecast': forecast_data,
            'start_datetime': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'weather_conditions': weather_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/insights')
def insights():
    """Get insights about the data"""
    try:
        train_df = pd.read_csv('train.csv')
        train_df['datetime'] = pd.to_datetime(train_df['datetime'])
        train_df['hour'] = train_df['datetime'].dt.hour
        train_df['dayofweek'] = train_df['datetime'].dt.dayofweek
        train_df['month'] = train_df['datetime'].dt.month
        
        # Calculate insights
        insights_data = {
            'total_rentals': int(float(train_df['count'].sum())),
            'average_daily_rentals': int(float(train_df['count'].mean())),
            'peak_hour': int(float(train_df.groupby('hour')['count'].mean().idxmax())),
            'peak_day': int(float(train_df.groupby('dayofweek')['count'].mean().idxmax())),
            'peak_month': int(float(train_df.groupby('month')['count'].mean().idxmax())),
            'weather_impact': {
                'clear': int(float(train_df[train_df['weather'] == 1]['count'].mean())),
                'mist': int(float(train_df[train_df['weather'] == 2]['count'].mean())),
                'light_rain': int(float(train_df[train_df['weather'] == 3]['count'].mean())),
                'heavy_rain': int(float(train_df[train_df['weather'] == 4]['count'].mean()))
            },
            'weekend_vs_weekday': {
                'weekend': int(float(train_df[train_df['dayofweek'].isin([5, 6])]['count'].mean())),
                'weekday': int(float(train_df[train_df['dayofweek'].isin([0, 1, 2, 3, 4])]['count'].mean()))
            }
        }
        
        return jsonify(insights_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Bike Sharing Demand Flask App...")
    
    # Load model and scalers
    if load_model_and_scalers():
        print("Model loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check model files.")
