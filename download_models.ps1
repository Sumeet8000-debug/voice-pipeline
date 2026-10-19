# Rehydrates Pocket TTS model files. Run once after cloning.
$ErrorActionPreference = "Stop"
$base = "https://huggingface.co/theoracleguy/pocket-tts-onnx/resolve/main/onnx/english_2026-04"
$out  = "onnx\english_2026-04"

if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out -Force | Out-Null }

$files = @{
    "flow_lm_main_int8.onnx"     = "$base/flow_lm_main_int8.onnx"
    "flow_lm_flow_int8.onnx"     = "$base/flow_lm_flow_int8.onnx"
    "mimi_decoder_int8.onnx"     = "$base/mimi_decoder_int8.onnx"
    "mimi_encoder_int8.onnx"     = "$base/mimi_encoder_int8.onnx"
    "text_conditioner_int8.onnx" = "$base/text_conditioner_int8.onnx"
}

foreach ($name in $files.Keys) {
    $dest = Join-Path $out $name
    if (Test-Path $dest) { Write-Host "skip $name (exists)"; continue }
    Write-Host "downloading $name ..."
    curl.exe -L -s -o $dest $files[$name]
    Write-Host "  -> $((Get-Item $dest).Length) bytes"
}

# ONNX Runtime DLL (needed for local dev)
$ortDir = "onnxruntime-win-x64-1.31.0"
if (-not (Test-Path "$ortDir\lib\onnxruntime.dll")) {
    Write-Host "downloading ONNX Runtime 1.31.0 ..."
    curl.exe -L -s -o ort131.zip "https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-win-x64-1.31.0.zip"
    Expand-Archive -Path ort131.zip -DestinationPath . -Force
    Copy-Item "$ortDir\lib\onnxruntime.dll" . -Force
    Remove-Item ort131.zip -Force
}

Write-Host "`nDone. Models are in onnx\english_2026-04\"
