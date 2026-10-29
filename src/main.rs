use anyhow::Result;
use ndarray::{Array, Array2, ArrayD, IxDyn};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{DynValue, Value};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

const BASE: &str = "onnx/english_2026-04";

#[derive(Debug, Deserialize)]
struct StateEntry {
    dtype: String,
    fill: String,
    input_name: String,
    output_name: String,
    module: String,
    key: String,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct Bundle {
    flow_lm_state_manifest: Vec<StateEntry>,
    mimi_state_manifest: Vec<StateEntry>,
    latent_dim: usize,
    conditioning_dim: usize,
    sample_rate: usize,
    tokenizer_file: String,
    model_recommended_frames_after_eos: Option<usize>,
    #[allow(dead_code)] samples_per_frame: usize,
    #[allow(dead_code)] frame_rate: f32,
}

enum StateData {
    F32(ArrayD<f32>),
    I64(ArrayD<i64>),
    Bool(ArrayD<bool>),
}

fn filled_array(shape: &[usize], dtype: &str, fill: &str) -> StateData {
    let shape = IxDyn(shape);
    match dtype {
        "float32" => StateData::F32(if fill == "nan" {
            ArrayD::from_elem(shape, f32::NAN)
        } else if fill == "ones" {
            ArrayD::from_elem(shape, 1.0f32)
        } else {
            ArrayD::zeros(shape)
        }),
        "int64" => StateData::I64(ArrayD::zeros(shape)),
        "bool" => StateData::Bool(ArrayD::from_elem(shape, fill == "ones")),
        _ => unreachable!(),
    }
}

fn init_state(manifest: &[StateEntry]) -> HashMap<String, StateData> {
    let mut state = HashMap::new();
    for e in manifest {
        state.insert(e.input_name.clone(), filled_array(&e.shape, &e.dtype, &e.fill));
    }
    state
}

fn copy_into<T: Clone>(src: &ArrayD<T>, target: &mut ArrayD<T>) {
    let src_shape = src.shape().to_vec();
    let src_size: usize = src_shape.iter().product();
    for flat in 0..src_size {
        let mut idx = vec![0usize; src_shape.len()];
        let mut rem = flat;
        for i in (0..src_shape.len()).rev() {
            idx[i] = rem % src_shape[i];
            rem /= src_shape[i];
        }
        target[IxDyn(&idx)] = src[IxDyn(&idx)].clone();
    }
}

fn adapt_f32(src: ArrayD<f32>, entry: &StateEntry) -> StateData {
    if src.shape() == entry.shape.as_slice() {
        return StateData::F32(src);
    }
    let target_size: usize = entry.shape.iter().product();
    if src.len() == target_size {
        if let Ok(r) = src.clone().into_shape_with_order(IxDyn(&entry.shape)) {
            return StateData::F32(r);
        }
    }
    let mut target = match filled_array(&entry.shape, "float32", &entry.fill) {
        StateData::F32(a) => a,
        _ => unreachable!(),
    };
    if src.ndim() == entry.shape.len() {
        copy_into(&src, &mut target);
    }
    StateData::F32(target)
}

fn adapt_i64(src: ArrayD<i64>, entry: &StateEntry) -> StateData {
    if src.shape() == entry.shape.as_slice() {
        return StateData::I64(src);
    }
    let target_size: usize = entry.shape.iter().product();
    if src.len() == target_size {
        if let Ok(r) = src.clone().into_shape_with_order(IxDyn(&entry.shape)) {
            return StateData::I64(r);
        }
    }
    let mut target = match filled_array(&entry.shape, "int64", &entry.fill) {
        StateData::I64(a) => a,
        _ => unreachable!(),
    };
    if src.ndim() == entry.shape.len() {
        copy_into(&src, &mut target);
    }
    StateData::I64(target)
}

fn apply_alba(state: &mut HashMap<String, StateData>, manifest: &[StateEntry]) -> Result<()> {
    let data = fs::read(format!("{}/alba.safetensors", BASE))?;
    let st = SafeTensors::deserialize(&data).map_err(|e| anyhow::anyhow!("safetensors: {:?}", e))?;
    for entry in manifest {
        let key = format!("{}/{}", entry.module, entry.key);
        let tensor = match st.tensor(&key) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let shape: Vec<usize> = tensor.shape().to_vec();
        match entry.dtype.as_str() {
            "float32" => {
                let vals: Vec<f32> = tensor.data().chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let arr = Array::<f32, _>::from_shape_vec(IxDyn(&shape), vals)
                    .map_err(|e| anyhow::anyhow!("shape: {:?}", e))?;
                state.insert(entry.input_name.clone(), adapt_f32(arr, entry));
            }
            "int64" => {
                let vals: Vec<i64> = tensor.data().chunks_exact(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
                let arr = Array::<i64, _>::from_shape_vec(IxDyn(&shape), vals)
                    .map_err(|e| anyhow::anyhow!("shape: {:?}", e))?;
                state.insert(entry.input_name.clone(), adapt_i64(arr, entry));
            }
            _ => {}
        }
    }
    Ok(())
}

fn state_to_dyn(d: &StateData) -> Result<DynValue> {
    Ok(match d {
        StateData::F32(a) => Value::from_array(a.clone())?.into_dyn(),
        StateData::I64(a) => Value::from_array(a.clone())?.into_dyn(),
        StateData::Bool(a) => Value::from_array(a.clone())?.into_dyn(),
    })
}

macro_rules! load_session {
    ($env:expr, $name:expr) => {{
        let path = format!("{}/{}", BASE, $name);
        Session::builder($env)
            .map_err(|e| anyhow::anyhow!("builder {}: {}", path, e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("opt {}: {}", path, e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("threads {}: {}", path, e))?
            .commit_from_file(&path)
            .map_err(|e| anyhow::anyhow!("commit {}: {}", path, e))?
    }};
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64) / (1u64 << 31) as f64
    }
    fn normal(&mut self, mean: f64, std: f64) -> f32 {
        let u1 = self.next().max(1e-10);
        let u2 = self.next();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        (mean + std * z) as f32
    }
}

fn main() -> Result<()> {
    let env = ort::init().with_name("pocket-tts").build()?;
    let bundle: Bundle = serde_json::from_str(&fs::read_to_string(format!("{}/bundle.json", BASE))?)?;

    let sp = sentencepiece::SentencePieceProcessor::open(format!("{}/{}", BASE, bundle.tokenizer_file))
        .map_err(|e| anyhow::anyhow!("tokenizer: {:?}", e))?;
    let text = "Hello from Pocket TTS. This is the voice pipeline.";
    let ids: Vec<i64> = sp.encode(text)
        .map_err(|e| anyhow::anyhow!("encode: {:?}", e))?
        .iter().map(|p| p.id as i64).collect();
    println!("Tokens: {}", ids.len());

    let mut text_cond = load_session!(&env, "text_conditioner_int8.onnx");
    let n = ids.len();
    let tokens: Array2<i64> = Array2::from_shape_fn((1, n), |(_, j)| ids[j]);
    let tokens_v: DynValue = Value::from_array(tokens.into_dyn())?.into_dyn();
    let out = text_cond.run(ort::inputs!["token_ids" => tokens_v])?;
    let text_emb: ArrayD<f32> = out.get("embeddings")
        .ok_or_else(|| anyhow::anyhow!("no embeddings"))?
        .try_extract_array::<f32>()?.to_owned();
    drop(out);
    println!("text_emb: {:?}", text_emb.shape());

    let mut lm_state = init_state(&bundle.flow_lm_state_manifest);
    apply_alba(&mut lm_state, &bundle.flow_lm_state_manifest)?;
    println!("LM state: {} (alba applied)", lm_state.len());

    let mut mimi_state = init_state(&bundle.mimi_state_manifest);
    println!("Mimi state: {}", mimi_state.len());

    let mut lm_main = load_session!(&env, "flow_lm_main_int8.onnx");
    let mut lm_flow = load_session!(&env, "flow_lm_flow_int8.onnx");
    let mut decoder = load_session!(&env, "mimi_decoder_int8.onnx");

    let empty_seq = ArrayD::<f32>::zeros(IxDyn(&[1, 0, bundle.latent_dim]));
    let empty_seq_v: DynValue = Value::from_array(empty_seq)?.into_dyn();
    let text_emb_v: DynValue = Value::from_array(text_emb.clone())?.into_dyn();
    let mut inputs: HashMap<String, DynValue> = HashMap::new();
    inputs.insert("sequence".into(), empty_seq_v);
    inputs.insert("text_embeddings".into(), text_emb_v);
    for (k, v) in &lm_state { inputs.insert(k.clone(), state_to_dyn(v)?); }

    println!("Prefill...");
    let outputs = lm_main.run(inputs)?;
    let prefill_cond = outputs.get("conditioning").unwrap().try_extract_array::<f32>()?.to_owned();
    let prefill_eos = outputs.get("eos_logit").unwrap().try_extract_array::<f32>()?.to_owned();
    for e in &bundle.flow_lm_state_manifest {
        if let Some(out) = outputs.get(&e.output_name) {
            let d = match e.dtype.as_str() {
                "float32" => StateData::F32(out.try_extract_array::<f32>()?.to_owned()),
                "int64" => StateData::I64(out.try_extract_array::<i64>()?.to_owned()),
                "bool" => StateData::Bool(out.try_extract_array::<bool>()?.to_owned()),
                _ => continue,
            };
            lm_state.insert(e.input_name.clone(), d);
        }
    }
    drop(outputs);
    let mag: f32 = prefill_cond.iter().map(|x| x.abs()).sum::<f32>() / prefill_cond.len() as f32;
    println!("Prefill: eos={:.4}, cond_mag={:.4}", prefill_eos[[0,0]], mag);

    let empty_text = ArrayD::<f32>::zeros(IxDyn(&[1, 0, bundle.conditioning_dim]));
    let lsd_steps = 4usize;
    let dt = 1.0f32 / lsd_steps as f32;
    let st_buffers: Vec<(ArrayD<f32>, ArrayD<f32>)> = (0..lsd_steps)
        .map(|j| {
            let s = j as f32 / lsd_steps as f32;
            let t = s + dt;
            (ArrayD::from_elem(IxDyn(&[1, 1]), s),
             ArrayD::from_elem(IxDyn(&[1, 1]), t))
        }).collect();

    let temperature = 0.7f64;
    let std_dev = temperature.sqrt();
    let mut rng = Rng::new(0xDEADBEEF);
    let mut latents: Vec<ArrayD<f32>> = Vec::new();
    let mut curr: ArrayD<f32> = ArrayD::from_elem(IxDyn(&[1, 1, bundle.latent_dim]), f32::NAN);
    let frames_after_eos = bundle.model_recommended_frames_after_eos.unwrap_or(5);
    let max_frames = 200usize;
    let mut eos_step: Option<usize> = None;

    for step in 0..max_frames {
        let curr_v: DynValue = Value::from_array(curr.clone())?.into_dyn();
        let empty_text_v: DynValue = Value::from_array(empty_text.clone())?.into_dyn();
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert("sequence".into(), curr_v);
        inputs.insert("text_embeddings".into(), empty_text_v);
        for (k, v) in &lm_state { inputs.insert(k.clone(), state_to_dyn(v)?); }

        let outputs = lm_main.run(inputs)?;
        let cond = outputs.get("conditioning").unwrap().try_extract_array::<f32>()?.to_owned();
        let eos = outputs.get("eos_logit").unwrap().try_extract_array::<f32>()?.to_owned();
        for e in &bundle.flow_lm_state_manifest {
            if let Some(out) = outputs.get(&e.output_name) {
                let d = match e.dtype.as_str() {
                    "float32" => StateData::F32(out.try_extract_array::<f32>()?.to_owned()),
                    "int64" => StateData::I64(out.try_extract_array::<i64>()?.to_owned()),
                    "bool" => StateData::Bool(out.try_extract_array::<bool>()?.to_owned()),
                    _ => continue,
                };
                lm_state.insert(e.input_name.clone(), d);
            }
        }
        drop(outputs);

        let eos_val = eos[[0, 0]];
        if step < 5 || step % 20 == 0 { println!("Step {}: eos={:.3}", step, eos_val); }

        if eos_val > -4.0 && eos_step.is_none() {
            eos_step = Some(step);
            println!("EOS at step {}", step);
        }
        if let Some(es) = eos_step { if step >= es + frames_after_eos { break; } }

        let mut x: ArrayD<f32> = ArrayD::from_shape_fn(
            IxDyn(&[1, bundle.latent_dim]),
            |_| rng.normal(0.0, std_dev),
        );
        for (s, t) in &st_buffers {
            let c_v: DynValue = Value::from_array(cond.clone())?.into_dyn();
            let s_v: DynValue = Value::from_array(s.clone())?.into_dyn();
            let t_v: DynValue = Value::from_array(t.clone())?.into_dyn();
            let x_v: DynValue = Value::from_array(x.clone())?.into_dyn();
            let flow_out = lm_flow.run(ort::inputs![
                "c" => c_v, "s" => s_v, "t" => t_v, "x" => x_v
            ])?;
            let flow: ArrayD<f32> = flow_out.get("flow_dir")
                .ok_or_else(|| anyhow::anyhow!("no flow_dir"))?
                .try_extract_array::<f32>()?.to_owned();
            let scaled = &flow * dt;
            x = &x + &scaled;
        }
        let latent = x.into_shape_with_order(IxDyn(&[1, 1, bundle.latent_dim]))?;
        latents.push(latent.clone());
        curr = latent;
    }
    println!("Generated {} latents", latents.len());

    let chunk_size = 15;
    let mut audio: Vec<f32> = Vec::new();
    for start in (0..latents.len()).step_by(chunk_size) {
        let end = (start + chunk_size).min(latents.len());
        let seq_len = end - start;
        let mut latent_seq = ArrayD::<f32>::zeros(IxDyn(&[1, seq_len, bundle.latent_dim]));
        for (i, l) in latents[start..end].iter().enumerate() {
            for j in 0..bundle.latent_dim { latent_seq[[0, i, j]] = l[[0, 0, j]]; }
        }
        let latent_v: DynValue = Value::from_array(latent_seq)?.into_dyn();
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert("latent".into(), latent_v);
        for (k, v) in &mimi_state { inputs.insert(k.clone(), state_to_dyn(v)?); }

        let outputs = decoder.run(inputs)?;
        let audio_out: ArrayD<f32> = outputs.get("audio_frame")
            .ok_or_else(|| anyhow::anyhow!("no audio_frame"))?
            .try_extract_array::<f32>()?.to_owned();
        for v in audio_out.iter() { audio.push(*v); }
        for e in &bundle.mimi_state_manifest {
            if let Some(out) = outputs.get(&e.output_name) {
                let d = match e.dtype.as_str() {
                    "float32" => StateData::F32(out.try_extract_array::<f32>()?.to_owned()),
                    "int64" => StateData::I64(out.try_extract_array::<i64>()?.to_owned()),
                    "bool" => StateData::Bool(out.try_extract_array::<bool>()?.to_owned()),
                    _ => continue,
                };
                mimi_state.insert(e.input_name.clone(), d);
            }
        }
    }
    println!("Audio: {} samples ({:.2}s @ {} Hz)",
        audio.len(), audio.len() as f32 / bundle.sample_rate as f32, bundle.sample_rate);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: bundle.sample_rate as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("rust_output.wav", spec)?;
    for s in &audio { writer.write_sample(*s)?; }
    writer.finalize()?;
    println!("Saved rust_output.wav");
    Ok(())
}
