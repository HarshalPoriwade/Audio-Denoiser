import tensorflow as tf
import os
import random

def get_audio_paths(directory, exclude_speech=False):
    """Recursively find all .wav files in a directory."""
    paths = []
    if not os.path.exists(directory):
        return paths
        
    for root, _, files in os.walk(directory):
        if exclude_speech and 'speech' in root.lower():
            continue
        for f in files:
            if f.endswith('.wav'):
                paths.append(os.path.join(root, f))
    return paths

def load_and_resample(file_path, target_sr=16000):
    """Loads an audio file and resamples it to 16kHz."""
    audio_binary = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    
    if sr != target_sr:
        # Resample audio
        audio = tf.numpy_function(
            lambda a, s: __import__('librosa').resample(a, orig_sr=s, target_sr=target_sr),
            [audio, sr], tf.float32
        )
    return audio

def create_lstm_dataset(clean_dir, noise_dirs, batch_size=32, block_len=512):
    """
    Builds the tf.data.Dataset for training.
    Yields overlapping blocks of audio data.
    """
    clean_paths = get_audio_paths(clean_dir)
    
    noise_paths = []
    for d in noise_dirs:
        noise_paths.extend(get_audio_paths(d, exclude_speech=True))
        
    def generate_data():
        while True:
            c_path = random.choice(clean_paths)
            n_path = random.choice(noise_paths)
            
            clean_audio = load_and_resample(c_path)
            noise_audio = load_and_resample(n_path)
            
            # Match lengths
            min_len = tf.minimum(tf.shape(clean_audio)[0], tf.shape(noise_audio)[0])
            clean_audio = clean_audio[:min_len]
            noise_audio = noise_audio[:min_len]
            
            # Mix signals with random SNR
            snr = tf.random.uniform([], 0.1, 0.9)
            mixed_audio = (clean_audio * snr) + (noise_audio * (1.0 - snr))
            
            # Normalize
            max_val = tf.maximum(tf.reduce_max(tf.abs(mixed_audio)), 1e-6)
            mixed_audio = mixed_audio / max_val
            clean_audio = clean_audio / max_val
            
            # Chunk into 512-sample blocks
            num_blocks = min_len // block_len
            if num_blocks == 0:
                continue
                
            mixed_blocks = tf.reshape(mixed_audio[:num_blocks * block_len], (num_blocks, block_len, 1))
            clean_blocks = tf.reshape(clean_audio[:num_blocks * block_len], (num_blocks, block_len, 1))
            
            yield mixed_blocks, clean_blocks

    dataset = tf.data.Dataset.from_generator(
        generate_data,
        output_signature=(
            tf.TensorSpec(shape=(None, block_len, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, block_len, 1), dtype=tf.float32)
        )
    )
    
    # Unbatch sequences into blocks and re-batch
    dataset = dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
