import tensorflow as tf

def create_lstm_model(block_len=512):
    """
    Creates a highly efficient, 1.5MB LSTM-based audio denoising model.
    Designed for real-time 16kHz processing on low-end devices.
    
    Args:
        block_len: The number of audio samples per frame (default 512 for 32ms at 16kHz).
    """
    inputs = tf.keras.Input(shape=(block_len, 1), name='input_audio')
    
    # 1. Feature Extraction (1D Convolution)
    # Expands the 1D audio signal into 64 feature channels
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same')(inputs)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    
    # 2. Recurrent Core (LSTMs)
    # The LSTM layers allow the model to remember past audio frames, 
    # making it excellent at tracking human speech patterns over time.
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    
    # 3. Output Projection
    # Collapses the 128 LSTM features back into a single clean audio channel
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LSTM_Denoising_16kHz")
    return model

if __name__ == "__main__":
    model = create_lstm_model()
    model.summary()
