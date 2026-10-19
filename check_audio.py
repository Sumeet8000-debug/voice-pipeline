import numpy as np, soundfile as sf
audio, sr = sf.read("reference_out.wav")
print(f"sample rate: {sr}")
print(f"shape: {audio.shape}, dtype: {audio.dtype}")
print(f"min={audio.min():.6f}, max={audio.max():.6f}")
print(f"mean={audio.mean():.6f}, std={audio.std():.6f}")
print(f"max_abs={np.abs(audio).max():.6f}")
print(f"first 20 samples: {audio[:20]}")
