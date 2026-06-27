import tensorflow as tf
import librosa
import soundfile as sf
import numpy as np
import os
from model_lstm import create_lstm_model

def test_lstm_model(input_wav_path, output_wav_path, weights_path='lstm_best.weights.h5'):
    """
    Loads the LSTM model, processes the input audio block-by-block,
    and saves the denoised output.
    """
    print(f"Loading input audio: {input_wav_path}")
    audio, sr = librosa.load(input_wav_path, sr=16000)
    
    print("Loading 1.5MB LSTM Model...")
    model = create_lstm_model(block_len=512)
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        print(f"Warning: {weights_path} not found. Output will be noisy.")
        
    # Pad audio to block length multiple
    block_len = 512
    pad_len = block_len - (len(audio) % block_len)
    audio_padded = np.pad(audio, (0, pad_len), mode='constant')
    
    # Block-by-block processing
    print("Denoising audio...")
    num_blocks = len(audio_padded) // block_len
    audio_blocks = audio_padded.reshape((num_blocks, block_len, 1))
    
    # Run inference
    clean_blocks = model.predict(audio_blocks, batch_size=32)
    
    # Flatten blocks to 1D array
    clean_audio = clean_blocks.flatten()
    
    # Remove padding
    clean_audio = clean_audio[:len(audio)]
    
    # Normalize output to prevent clipping
    max_val = np.max(np.abs(clean_audio))
    if max_val > 1.0:
        clean_audio = clean_audio / max_val
    
    print(f"Saving clean output to: {output_wav_path}")
    sf.write(output_wav_path, clean_audio, 16000)
    print("Done!")

if __name__ == "__main__":
    # Example usage:
    # test_lstm_model("noisy_input.wav", "clean_test.wav")
    pass
