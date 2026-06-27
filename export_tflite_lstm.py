import tensorflow as tf
from model_lstm import create_lstm_model

def export_to_tflite(weights_path='lstm_best.weights.h5', output_path='lstm_float16.tflite'):
    """
    Exports the trained LSTM model to TensorFlow Lite format
    with optimizations for edge deployment.
    """
    print(f"Loading 1.5MB LSTM Model architecture...")
    model = create_lstm_model(block_len=512)
    
    try:
        model.load_weights(weights_path)
        print(f"Successfully loaded weights from {weights_path}")
    except Exception as e:
        print(f"Warning: Could not load weights. Exporting raw architecture. Error: {e}")

    print("Initializing TFLite Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply default optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Apply Float16 quantization
    converter.target_spec.supported_types = [tf.float16]
    
    print("Converting model (this may take a moment)...")
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"✅ Success! Edge-ready model exported to: {output_path}")

if __name__ == "__main__":
    export_to_tflite()
