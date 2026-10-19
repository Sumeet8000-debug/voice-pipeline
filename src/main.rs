use anyhow::Result;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{DynValue, Value};
use ndarray::{Array2, ArrayD, IxDyn};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

const BASE: &str = "onnx/english_2026-04";

#[derive(Debug, Deserialize)]
struct StateEntry {
    dtype: String,
    fill: String,
    #[allow(dead_code)]
    index: usize,
    input_name: String,
    #[allow(dead_code)]
    output_name: String,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct Bundle {
    flow_lm_state_manifest: Vec<StateEntry>,
    mimi_state_manifest: Vec<StateEntry>,
    latent_dim: usize,
    conditioning_dim: usize,
    sample_rate: usize,
    samples_per_frame: usize,
    frame_rate: f32,
    tokenizer_file: String,
}

fn init_tensor(entry: &StateEntry) -> Result<DynValue> {
    let shape = IxDyn(&entry.shape);
    let value: DynValue = match entry.dtype.as_str() {
        "float32" => {
            let data: ArrayD<f32> = if entry.fill == "nan" {
                ArrayD::from_elem(shape, f32::NAN)
            } else if entry.fill == "ones" {
                ArrayD::from_elem(shape, 1.0f32)
            } else {
                ArrayD::zeros(shape)
            };
            Value::from_array(data)?.into_dyn()
        }
        "int64" => {
            let data: ArrayD<i64> = ArrayD::zeros(shape);
            Value::from_array(data)?.into_dyn()
        }
        "bool" => {
            let data: ArrayD<bool> = ArrayD::from_elem(shape, entry.fill == "ones");
            Value::from_array(data)?.into_dyn()
        }
        other => anyhow::bail!("Unknown dtype: {}", other),
    };
    Ok(value)
}

fn main() -> Result<()> {
    let env = ort::init().with_name("pocket-tts").build()?;

    let bundle_json = fs::read_to_string(format!("{}/bundle.json", BASE))?;
    let bundle: Bundle = serde_json::from_str(&bundle_json)?;

    println!("=== Bundle metadata ===");
    println!("latent_dim = {}", bundle.latent_dim);
    println!("conditioning_dim = {}", bundle.conditioning_dim);
    println!("sample_rate = {}", bundle.sample_rate);
    println!("samples_per_frame = {}", bundle.samples_per_frame);
    println!("frame_rate = {}", bundle.frame_rate);
    println!("flow_lm_states = {}", bundle.flow_lm_state_manifest.len());
    println!("mimi_states = {}", bundle.mimi_state_manifest.len());

    // Tokenizer
    let sp = sentencepiece::SentencePieceProcessor::open(format!(
        "{}/{}",
        BASE, bundle.tokenizer_file
    ))
    .map_err(|e| anyhow::anyhow!("tokenizer open: {:?}", e))?;

    let text = "Hello from Pocket TTS. This is the voice pipeline.";
    let pieces = sp
        .encode(text)
        .map_err(|e| anyhow::anyhow!("encode: {:?}", e))?;
    let ids: Vec<i64> = pieces.iter().map(|p| p.id as i64).collect();

    println!("\n=== Tokenization ===");
    println!("text: {}", text);
    println!("token count: {}", ids.len());
    println!("token_ids: {:?}", ids);

    // text_conditioner
    let mut text_cond = Session::builder(&env)
        .map_err(|e| anyhow::anyhow!("builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("opt: {e}"))?
        .with_intra_threads(4)
        .map_err(|e| anyhow::anyhow!("threads: {e}"))?
        .commit_from_file(format!("{}/text_conditioner_int8.onnx", BASE))
        .map_err(|e| anyhow::anyhow!("commit: {e}"))?;

    let n = ids.len();
    let arr: Array2<i64> = Array2::from_shape_fn((1, n), |(_, j)| ids[j]);
    let tokens_val: DynValue = Value::from_array(arr.into_dyn())?.into_dyn();

    let outputs = text_cond.run(ort::inputs!["token_ids" => tokens_val])?;

    if let Some(emb) = outputs.get("embeddings") {
        let view = emb.try_extract_array::<f32>()?;
        println!("\n=== text_conditioner output ===");
        println!("text_embeddings shape: {:?}", view.shape());
    } else {
        println!("\nWARN: no 'embeddings' output. Available:");
        for (name, _) in outputs.iter() {
            println!("  {}", name);
        }
    }

    // State init
    let mut lm_state: HashMap<String, DynValue> = HashMap::new();
    for e in &bundle.flow_lm_state_manifest {
        lm_state.insert(e.input_name.clone(), init_tensor(e)?);
    }
    println!("\n=== State tensors ===");
    println!("LM states initialized: {}", lm_state.len());

    let mut mimi_state: HashMap<String, DynValue> = HashMap::new();
    for e in &bundle.mimi_state_manifest {
        mimi_state.insert(e.input_name.clone(), init_tensor(e)?);
    }
    println!("Mimi states initialized: {}", mimi_state.len());

    Ok(())
}
