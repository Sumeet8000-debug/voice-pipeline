---
license: cc-by-4.0
language:
- en
tags:
- text-to-speech
- onnx
- onnxruntime
- streaming
- android
- edge
base_model: kyutai/pocket-tts-without-voice-cloning
library_name: onnxruntime
pipeline_tag: text-to-speech
---

# Pocket TTS 100M ONNX INT8

CPU-oriented, streaming ONNX export of
[Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts), packaged for
on-device voice assistants. The 120.3 MiB bundle splits the stateful model into
five graphs so a runtime can generate and play one 80 ms audio frame before
computing the next one.

This repository deliberately uses Kyutai's public
[`pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning)
checkpoint and bakes in its `alba` preset. It is English-only and
does **not** support arbitrary voice cloning.

## Files

| File | Purpose | Precision |
|---|---|---|
| `lm_main.int8.onnx` | Autoregressive backbone, EOS and recurrent state | dynamic MatMul INT8 |
| `lm_flow.int8.onnx` | Four-step flow-matching update | dynamic MatMul INT8 |
| `decoder.int8.onnx` | Streaming Mimi latent-to-audio decoder | dynamic MatMul INT8 |
| `text_conditioner.onnx` | Token IDs to LM embeddings | float32 |
| `encoder.onnx` | Fixed `alba` voice-state adapter | float32 |
| `vocab.json`, `token_scores.json` | SentencePiece-compatible runtime tables | JSON |
| `tokenizer.model` | Original pinned tokenizer | SentencePiece |

All audio is mono Float32 PCM at 24 kHz. Each decoder invocation returns 1,920
samples (80 ms). `manifest.json` records exact source/exporter revisions,
opsets, byte sizes and SHA-256 checksums.

## Download

```bash
huggingface-cli download soniqo/Pocket-TTS-100M-ONNX-INT8 \
  --revision v1.0.0 --local-dir pocket-tts-onnx
```

## speech-core usage

[`soniqo/speech-core`](https://github.com/soniqo/speech-core) provides a C++17
ONNX Runtime backend that performs true frame interleaving:

```cpp
#include <speech_core/models/onnx_pocket_tts.h>

speech_core::PocketTtsConfig config;
config.intra_threads = 2;
config.flow_steps = 4;

speech_core::OnnxPocketTts tts("/models/pocket-tts-onnx", config);
tts.synthesize("Hello world.", "en",
    [](const float* samples, size_t count, bool final) {
        // Every non-final callback is one playable 80 ms frame.
    });
```

The five graphs are also compatible with
[`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx). Its current offline
Pocket wrapper buffers the latent sequence before decoder callbacks; use
speech-core when first-frame latency is the primary requirement.

## Galaxy S23 Ultra results

Measured on a Samsung SM-S918B, Snapdragon 8 Gen 2, Android 16, ONNX Runtime
1.27 CPU, two threads, four flow steps:

| Text | Warm TTFA p50 / p95 | Total p50 / p95 | RTF p50 | Peak RSS |
|---|---:|---:|---:|---:|
| `Hello world.` | 130 / 136 ms | 470 / 488 ms | 0.452 | 376.4 MiB |
| `I can call your contacts,` | 147 / 151 ms | 779 / 796 ms | 0.443 | 376.5 MiB |

The native backend emitted exactly one 1,920-sample frame per callback. On
Windows, native speech-core and sherpa-onnx using the same ONNX Runtime build
produced the same sample count and Float32 waveform hash for the fixed-seed
`Hello world.` test.

### Round-trip intelligibility

- Deployed Parakeet-EOU recognizer: three Pocket seeds, 138 syntheses, 6.79%
  micro WER, 2.91% CER, 103/138 exact, zero empty output and zero EOS failure.
- Assistant-response subset: 1.96% WER and 0.06% CER across 60 syntheses.
- Independent Faster-Whisper Large-v3 pass over all 46 untouched seed-123
  device WAVs: 1.43% WER, 0.40% CER and 42/46 exact after standard English
  normalization. Numbers and technical responses were exact.

Round-trip ASR measures intelligibility, not naturalness or listener
preference. Human listening remains the appropriate gate for voice quality and
prosody.

## Reproducibility

The export recipe is maintained in
[`soniqo/speech-models`](https://github.com/soniqo/speech-models/tree/main/models/pocket-tts/export).
It pins:

- source checkpoint `kyutai/pocket-tts-without-voice-cloning` at `d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3`;
- ONNX exporter `https://github.com/csukuangfj/pocket-tts-onnx-export.git` at `f075c00bf4bbfbb081a11fd99abbf39df3849e0c`;
- all Python/export dependencies in `pyproject.toml`.

A clean repack reproduced every non-manifest artifact byte-for-byte and all
five graphs passed full ONNX validation.

## Limitations

- English only and fixed `alba` voice; no arbitrary voice cloning.
- CPU optimized. NNAPI/QNN execution is not validated for these recurrent
  graphs.
- Dates, isolated digit sequences, uncommon proper names and tongue-twisters
  should be normalized or covered by product-specific regression tests.
- This model card does not claim a human MOS or preference result.

## License and attribution

The model checkpoint is CC BY 4.0. The baked-in voice was performed by
**Alba MacKenna** and is also released under CC BY 4.0; retain that
attribution in redistributed bundles and product notices. Voice source:
https://huggingface.co/kyutai/tts-voices#alba-mackenna.

Kyutai's implementation and the ONNX exporter code are MIT licensed. See
`LICENSE`, the upstream repositories, and `manifest.json` for the exact source
and artifact provenance.
