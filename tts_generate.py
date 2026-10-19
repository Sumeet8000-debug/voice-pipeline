import json
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import soundfile as sf

with open("bundle.json") as f:
    meta = json.load(f)

sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")
print(f"Tokenizer loaded. Vocab size: {sp.GetPieceSize()}")

providers = ["CPUExecutionProvider"]
text_cond = ort.InferenceSession("text_conditioner.onnx", providers=providers)
lm_main   = ort.InferenceSession("lm_main.int8.onnx", providers=providers)
lm_flow   = ort.InferenceSession("lm_flow.int8.onnx", providers=providers)
decoder   = ort.InferenceSession("decoder.int8.onnx", providers=providers)

def numpy_dtype(name):
    return {"float32": np.float32, "int64": np.int64, "bool": np.bool_}[name]

def init_state(manifest):
    state = {}
    for e in manifest:
        dt = numpy_dtype(e["dtype"])
        shape = tuple(e["shape"])
        fill = e["fill"]
        if fill == "nan":
            arr = np.full(shape, np.nan, dtype=dt)
        elif fill == "ones":
            arr = np.ones(shape, dtype=dt)
        else:
            arr = np.zeros(shape, dtype=dt)
        state[e["input_name"]] = arr
    return state

def update_state(state, result, manifest, offset):
    for e in manifest:
        state[e["input_name"]] = result[offset + e["index"]]

lm_manifest = meta["flow_lm_state_manifest"]
dec_manifest = meta["mimi_state_manifest"]
latent_dim = meta["latent_dim"]

lm_state = init_state(lm_manifest)
dec_state = init_state(dec_manifest)

text = "Hello from Pocket TTS. This is the voice pipeline."
token_ids = sp.EncodeAsIds(text)
print(f"Token IDs: {token_ids}")
token_ids_arr = np.array([token_ids], dtype=np.int64)
text_emb = text_cond.run(None, {"token_ids": token_ids_arr})[0]
if text_emb.ndim == 2:
    text_emb = text_emb[None]
print(f"Text embeddings shape: {text_emb.shape}")

# Prefill
empty_seq = np.zeros((1, 0, latent_dim), dtype=np.float32)
result = lm_main.run(None, {"sequence": empty_seq, "text_embeddings": text_emb, **lm_state})
conditioning = result[0]
eos_logit = result[1]
update_state(lm_state, result, lm_manifest, 2)
print(f"Prefill: eos_logit={eos_logit[0,0]:.3f}, cond_mag={np.abs(conditioning).mean():.3f}")

# Generation loop
latents = []
eos_step = None
frames_after_eos = 3
max_frames = 200
min_frames = 15

for step in range(max_frames):
    # Flow-match current conditioning -> new latent
    x = np.random.randn(1, latent_dim).astype(np.float32)
    for k in range(4):
        s = np.array([[k / 4.0]], dtype=np.float32)
        t = np.array([[(k + 1) / 4.0]], dtype=np.float32)
        flow = lm_flow.run(None, {"c": conditioning, "s": s, "t": t, "x": x})[0]
        x = x + flow * 0.25
    latent = x.reshape(1, 1, latent_dim)
    latents.append(latent)

    # EOS check on PREVIOUS lm_main's logit
    if step >= min_frames and eos_logit[0][0] > -4.0 and eos_step is None:
        eos_step = step
        print(f"EOS at step {step}")
    if eos_step is not None and step >= eos_step + frames_after_eos:
        break

    # Next LM call with GROWING sequence (all latents so far)
    seq = np.concatenate(latents, axis=1)
    result = lm_main.run(None, {"sequence": seq, "text_embeddings": text_emb, **lm_state})
    conditioning = result[0]
    eos_logit = result[1]
    update_state(lm_state, result, lm_manifest, 2)

    if step < 10 or step % 20 == 0:
        print(f"Step {step}: eos={eos_logit[0,0]:.3f}, latent_mag={np.abs(latent).mean():.3f}")

print(f"Generated {len(latents)} latents")

# Decode
chunk_size = 15
audio_chunks = []
for start in range(0, len(latents), chunk_size):
    chunk = np.concatenate(latents[start:start+chunk_size], axis=1)
    result = decoder.run(None, {"latent": chunk, **dec_state})
    audio_chunks.append(result[0].reshape(-1))
    update_state(dec_state, result, dec_manifest, 1)

audio = np.concatenate(audio_chunks)
print(f"Audio shape: {audio.shape} ({len(audio)/24000:.2f}s)")
sf.write("first_speech.wav", audio, 24000)
print("Saved first_speech.wav")
