import tensorflow as tf
import os
from dataset_lstm import create_lstm_dataset
from model_lstm import create_lstm_model

def si_sdr_loss(y_true, y_pred):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss."""
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)
    y_true = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
    y_pred = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)
    
    alpha = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True) / (tf.reduce_sum(y_true**2, axis=-1, keepdims=True) + 1e-8)
    target = alpha * y_true
    res = y_pred - target
    sdr = 10 * tf.experimental.numpy.log10(tf.reduce_sum(target**2, axis=-1) / (tf.reduce_sum(res**2, axis=-1) + 1e-8) + 1e-8)
    return -tf.reduce_mean(sdr)

def train_lstm():
    # Configure dataset paths
    CLEAN_DIR = './data/clean'
    NOISE_DIRS = ['./data/noise']
    
    print("Initializing 16kHz LSTM Dataset...")
    dataset = create_lstm_dataset(CLEAN_DIR, NOISE_DIRS, batch_size=64, block_len=512)
    
    print("Building 1.5MB LSTM Model...")
    model = create_lstm_model(block_len=512)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=si_sdr_loss
    )
    
    if os.path.exists('lstm_best.weights.h5'):
        print("Found existing LSTM weights, resuming...")
        model.load_weights('lstm_best.weights.h5')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'lstm_best.weights.h5',
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=12,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("Starting LSTM Training...")
    model.fit(
        dataset,
        steps_per_epoch=1000,
        epochs=100,
        callbacks=callbacks
    )

    model.save_weights('lstm_final.weights.h5')
    print("Training complete! Model saved as lstm_best.weights.h5 (Approx 1.5MB)")

if __name__ == "__main__":
    train_lstm()
