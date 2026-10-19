use anyhow::Result;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

fn main() -> Result<()> {
    // rc.14: build() returns Environment (not commit())
    let environment = ort::init().with_name("pocket-tts-prototype").build()?;

    // ort::Error<SessionBuilder> is not Send+Sync, so each builder step
    // needs .map_err(...) to convert to a string-based anyhow error.
    let mut builder = Session::builder(&environment)
        .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("optimization level: {e}"))?
        .with_intra_threads(4)
        .map_err(|e| anyhow::anyhow!("intra threads: {e}"))?;

    let session = builder
        .commit_from_file("onnx/english_2026-04/flow_lm_main_int8.onnx")
        .map_err(|e| anyhow::anyhow!("commit from file: {e}"))?;

    println!("Model loaded: flow_lm_main_int8.onnx");
    for i in session.inputs() {
        println!("  INPUT : {}  dtype={:?}", i.name(), i.dtype());
    }
    for o in session.outputs() {
        println!("  OUTPUT: {}  dtype={:?}", o.name(), o.dtype());
    }
    Ok(())
}
