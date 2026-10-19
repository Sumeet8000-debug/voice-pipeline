import onnxruntime as ort

for model_file in ["lm_main.int8.onnx", "encoder.onnx", "text_conditioner.onnx",
                   "lm_flow.int8.onnx", "decoder.int8.onnx"]:
    print(f"\n===== {model_file} =====")
    try:
        sess = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])
        print("INPUTS:")
        for inp in sess.get_inputs():
            print(f"  {inp.name}  shape={inp.shape}  type={inp.type}")
        print("OUTPUTS:")
        for out in sess.get_outputs():
            print(f"  {out.name}  shape={out.shape}  type={out.type}")
    except Exception as e:
        print(f"  FAILED: {e}")
