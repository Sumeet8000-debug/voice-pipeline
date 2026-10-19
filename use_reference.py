from reference import PocketTTSOnnx
from safetensors import safe_open
import soundfile as sf
import os

_local_voice = os.path.abspath("onnx/english_2026-04/alba.safetensors")

def _local_model_state(self, filename):
    result = {}
    with safe_open(_local_voice, framework="np") as handle:
        for key in handle.keys():
            module_name, tensor_key = key.split("/", 1)
            result.setdefault(module_name, {})
            result[module_name][tensor_key] = handle.get_tensor(key)
    return result

PocketTTSOnnx._hf_model_state = _local_model_state

tts = PocketTTSOnnx(
    models_dir="onnx",
    language="english_2026-04",
    precision="int8",
    device="cpu",
)

audio = tts.generate(
    "Hello from Pocket TTS. This is the voice pipeline.",
    voice="alba",
)

print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
print(f"Duration: {len(audio)/24000:.2f}s")

sf.write("reference_out.wav", audio, 24000)
print("Saved reference_out.wav")
