import tensorflow as tf

def create_lstm_model(block_len=512):
    """
    Builds the LSTM-based audio denoising model.
    Optimized for 16kHz audio processing on edge devices.
    
    Args:
        block_len (int): Samples per frame (default: 512).
    """
    inputs = tf.keras.Input(shape=(block_len, 1), name='input_audio')
    
    # Initial feature extraction
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same')(inputs)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    
    # Sequence modeling
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    
    # Output projection
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LSTM_Denoising_16kHz")
    return model

if __name__ == "__main__":
    model = create_lstm_model()
    model.summary()
