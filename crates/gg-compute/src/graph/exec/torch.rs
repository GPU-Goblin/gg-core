//! Native CUDA engine (prototype): the executor's own operators, run on LibTorch through Burn, with
//! PyTorch's automatic differentiation instead of Burn's. Burn computes every backward pass with
//! its own generic formulas, even on LibTorch (a depthwise convolution's weight gradient: 306 ms
//! against 3 ms); PyTorch's autograd calls cuDNN and ATen's dedicated backward kernels.
//!
//! The forward pass is the same code as every other engine (`apply`). Each trained parameter is
//! marked as requiring a gradient on the PyTorch side; LibTorch records the graph as Burn calls its
//! functions, and `backward` runs PyTorch's. A tensor modified in place while PyTorch still needs it
//! makes `backward` fail with an error, never a wrong gradient (PyTorch's version counters).
//!
//! Strict float32: cuDNN computes convolutions in TF32 by default, which put a ResNet18's gradients
//! 7 % away from the reference. The process must run with `NVIDIA_TF32_OVERRIDE=0` (set by the
//! executor and the validator before anything starts); the engine refuses to run otherwise.

use std::collections::BTreeMap;

use burn::backend::libtorch::{LibTorch, LibTorchDevice, TchTensor};
use burn::tensor::{Tensor, TensorPrimitive};
use gg_protocol::tensor_file::TensorF32;

use super::{
    Dyn, ExecError, ExecOptions, Prepared, StageBackward, StepResult, add_inner, apply,
    dataset_spec, dtype_err, dyn_bytes, env_bytes, from_tensor_f32, load_datasets_owned, loss_of,
    shape_err, stage_inputs, stage_param_specs, zeros_of,
};

type T = LibTorch<f32>;

fn map_tch(t: Dyn<T>, f: impl Fn(tch::Tensor) -> tch::Tensor) -> Dyn<T> {
    fn one<const D: usize>(
        x: Tensor<T, D>,
        f: &dyn Fn(tch::Tensor) -> tch::Tensor,
    ) -> Tensor<T, D> {
        let p = x.into_primitive().tensor();
        Tensor::from_primitive(TensorPrimitive::Float(TchTensor::new(f(p.tensor))))
    }
    match t {
        Dyn::F1(x) => Dyn::F1(one(x, &f)),
        Dyn::F2(x) => Dyn::F2(one(x, &f)),
        Dyn::F3(x) => Dyn::F3(one(x, &f)),
        Dyn::F4(x) => Dyn::F4(one(x, &f)),
        other => other,
    }
}

fn tch_of(t: &Dyn<T>) -> Option<tch::Tensor> {
    fn one<const D: usize>(x: &Tensor<T, D>) -> tch::Tensor {
        x.clone().into_primitive().tensor().tensor.shallow_clone()
    }
    match t {
        Dyn::F1(x) => Some(one(x)),
        Dyn::F2(x) => Some(one(x)),
        Dyn::F3(x) => Some(one(x)),
        Dyn::F4(x) => Some(one(x)),
        _ => None,
    }
}

/// A LibTorch tensor as the executor's value of rank `rank`.
fn dyn_of(x: tch::Tensor, rank: usize) -> Result<Dyn<T>, ExecError> {
    fn one<const D: usize>(x: tch::Tensor) -> Tensor<T, D> {
        Tensor::from_primitive(TensorPrimitive::Float(TchTensor::new(x)))
    }
    Ok(match rank {
        1 => Dyn::F1(one(x)),
        2 => Dyn::F2(one(x)),
        3 => Dyn::F3(one(x)),
        4 => Dyn::F4(one(x)),
        r => return Err(shape_err(format!("rank {r} tensor"))),
    })
}

/// A parameter straight onto the card, after the checks of every engine (declared shape, finite
/// values): one copy into LibTorch and one transfer. Burn's conversion made two more host copies,
/// which cost more than the transfer itself for SmolLM2 (measured: 200 against 90 ms for a stage).
fn upload_parameter(
    spec: &crate::graph::ParamSpec,
    t: &TensorF32,
    device: tch::Device,
) -> Result<Dyn<T>, ExecError> {
    if t.shape != spec.shape {
        return Err(shape_err(format!(
            "parameter '{}': shape {:?} provided, {:?} declared",
            spec.name, t.shape, spec.shape
        )));
    }
    t.check_finite()
        .map_err(|e| ExecError::Numeric(format!("parameter '{}': {e}", spec.name)))?;
    let shape: Vec<i64> = t.shape.iter().map(|d| *d as i64).collect();
    dyn_of(
        tch::Tensor::from_slice(&t.data)
            .reshape(&shape)
            .to_device(device),
        shape.len(),
    )
}

/// Gradient of a micro-batch on CUDA device `index`, with PyTorch's autograd.
pub fn gradient_cuda(
    index: usize,
    prepared: &Prepared,
    mut parameters: BTreeMap<String, TensorF32>,
    inputs: BTreeMap<String, TensorF32>,
    seed: u64,
    options: ExecOptions,
) -> Result<StepResult, ExecError> {
    crate::engine::cuda::check(index).map_err(|e| ExecError::Numeric(e.to_string()))?;
    if options.segments > 1 || options.offload_parameters {
        return gradient_cuda_segmented(
            index,
            prepared,
            parameters,
            inputs,
            seed,
            options.segments.max(1),
        );
    }
    let device = LibTorchDevice::Cuda(index);
    let graph = &prepared.graph;
    let mut env: BTreeMap<String, Dyn<T>> =
        load_datasets_owned::<T>(&device, graph, &prepared.index_limits, inputs)?;
    let mut tracked: Vec<(String, tch::Tensor)> = Vec::new();
    for p in &graph.parameters {
        let t = parameters
            .remove(&p.name)
            .ok_or_else(|| ExecError::Missing(p.name.clone()))?;
        let mut v = upload_parameter(p, &t, tch::Device::Cuda(index))?;
        drop(t);
        if p.trainable {
            v = map_tch(v, |t| t.set_requires_grad(true));
            if let Some(h) = tch_of(&v) {
                tracked.push((p.name.clone(), h));
            }
        }
        env.insert(p.name.clone(), v);
    }
    for name in &prepared.report.order {
        let node = &graph.nodes[prepared.by_output[name]];
        let args = node
            .inputs
            .iter()
            .map(|t| {
                env.get(t)
                    .cloned()
                    .ok_or_else(|| ExecError::Missing(t.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let out = apply(&device, node, args, seed)?;
        env.insert(node.output.clone(), out);
    }
    let peak_activation_bytes = env_bytes(&env);
    let loss = loss_of(graph, &env)?;
    let loss_t = tch_of(&Dyn::F1(loss)).expect("a float loss");
    let loss_value = loss_t.double_value(&[0]) as f32;
    if !loss_value.is_finite() {
        return Err(ExecError::Numeric(format!("non-finite loss: {loss_value}")));
    }
    loss_t
        .f_backward()
        .map_err(|e| ExecError::Numeric(format!("PyTorch backward: {e}")))?;
    drop(env);
    let mut gradients = BTreeMap::new();
    for (name, h) in tracked {
        let g = grad_of(&h, &name)?;
        gradients.insert(name, g);
    }
    Ok(StepResult {
        loss: loss_value,
        gradients,
        peak_activation_bytes,
        peak_parameter_bytes: 0,
    })
}

/// The gradient PyTorch accumulated on `h`, on the host with `h`'s shape; zeros when the loss does
/// not depend on it.
fn grad_of(h: &tch::Tensor, name: &str) -> Result<TensorF32, ExecError> {
    let dims: Vec<usize> = h.size().iter().map(|d| *d as usize).collect();
    let g = h.grad();
    let g = if g.defined() {
        // Straight into the host vector, without Burn's intermediate copies.
        let c = g.to_device(tch::Device::Cpu).contiguous().reshape([-1]);
        let n = c.numel();
        let mut data = vec![0f32; n];
        c.f_copy_data(&mut data, n)
            .map_err(|e| ExecError::Numeric(format!("gradient of '{name}': {e}")))?;
        TensorF32::new(dims.iter().map(|d| *d as u64).collect(), data)
            .map_err(|e| ExecError::Shape(e.to_string()))?
    } else {
        zeros_of(&dims)?
    };
    g.check_finite()
        .map_err(|e| ExecError::Numeric(format!("gradient of '{name}': {e}")))?;
    Ok(g)
}

/// Backward pass of a stage on CUDA device `index`, with PyTorch's autograd: the same inputs and
/// results as [`super::stage_backward`], which Burn's generic backward formulas compute. The
/// stage's trained parameters and its activation inputs are marked as requiring a gradient; the
/// last stage starts from the loss, another one from `Σ ⟨output, received gradient⟩`, the received
/// gradients of an output summed in the given order, as the generic path does.
pub fn stage_backward_cuda(
    index: usize,
    prepared: &Prepared,
    stage: &crate::graph::stages::StageSpec,
    parameters: &BTreeMap<String, TensorF32>,
    inputs: &BTreeMap<String, TensorF32>,
    output_gradients: &BTreeMap<String, Vec<TensorF32>>,
    seed: u64,
) -> Result<StageBackward, ExecError> {
    crate::engine::cuda::check(index).map_err(|e| ExecError::Numeric(e.to_string()))?;
    let device = LibTorchDevice::Cuda(index);
    let graph = &prepared.graph;
    let mut env: BTreeMap<String, Dyn<T>> =
        stage_inputs::<T>(&device, graph, &prepared.index_limits, stage, inputs)?;
    let mut tracked_inputs: Vec<(String, tch::Tensor)> = Vec::new();
    for name in &stage.inputs {
        if dataset_spec(graph, name).is_some() {
            continue;
        }
        let v = env
            .remove(name)
            .ok_or_else(|| ExecError::Missing(name.clone()))?;
        let v = map_tch(v, |t| t.set_requires_grad(true));
        let h = tch_of(&v).ok_or_else(|| dtype_err(format!("input '{name}' is not a float")))?;
        tracked_inputs.push((name.clone(), h));
        env.insert(name.clone(), v);
    }
    let mut tracked: Vec<(String, tch::Tensor)> = Vec::new();
    for p in stage_param_specs(graph, stage)? {
        let t = parameters
            .get(&p.name)
            .ok_or_else(|| ExecError::Missing(p.name.clone()))?;
        let mut v = upload_parameter(p, t, tch::Device::Cuda(index))?;
        if p.trainable {
            v = map_tch(v, |t| t.set_requires_grad(true));
            if let Some(h) = tch_of(&v) {
                tracked.push((p.name.clone(), h));
            }
        }
        env.insert(p.name.clone(), v);
    }
    for name in prepared
        .report
        .order
        .iter()
        .take(stage.end)
        .skip(stage.first)
    {
        let node = &graph.nodes[prepared.by_output[name]];
        let args = node
            .inputs
            .iter()
            .map(|t| {
                env.get(t)
                    .cloned()
                    .ok_or_else(|| ExecError::Missing(t.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let out = apply(&device, node, args, seed)?;
        env.insert(node.output.clone(), out);
    }
    let mut peak_bytes = env_bytes(&env);
    let mut loss = None;
    let start: Option<tch::Tensor> = if stage.last {
        let l = tch_of(&Dyn::F1(loss_of(graph, &env)?)).expect("a float loss");
        let v = l.double_value(&[0]) as f32;
        if !v.is_finite() {
            return Err(ExecError::Numeric(format!("non-finite loss: {v}")));
        }
        loss = Some(v);
        Some(l)
    } else {
        let mut acc: Option<tch::Tensor> = None;
        for name in &stage.outputs {
            let Some(list) = output_gradients.get(name).filter(|l| !l.is_empty()) else {
                continue;
            };
            let mut g = from_tensor_f32::<T>(&device, &list[0])?;
            for more in &list[1..] {
                g = add_inner(g, from_tensor_f32::<T>(&device, more)?)?;
            }
            peak_bytes += dyn_bytes(&g);
            let y = env
                .get(name)
                .ok_or_else(|| ExecError::Missing(name.clone()))?;
            if y.dims() != g.dims() {
                return Err(shape_err(format!(
                    "gradient received for '{name}' has shape {:?}, the output is {:?}",
                    g.dims(),
                    y.dims()
                )));
            }
            let (yt, gt) = (
                tch_of(y).ok_or_else(|| dtype_err("non-float stage output"))?,
                tch_of(&g).expect("a float gradient"),
            );
            let sum = (&yt * &gt).sum(tch::Kind::Float);
            acc = Some(match acc {
                Some(a) => a + sum,
                None => sum,
            });
        }
        acc
    };
    if let Some(start) = &start {
        start
            .f_backward()
            .map_err(|e| ExecError::Numeric(format!("PyTorch backward: {e}")))?;
    }
    drop(env);
    let mut gradients = BTreeMap::new();
    for (name, h) in &tracked {
        gradients.insert(name.clone(), grad_of(h, name)?);
    }
    let mut input_gradients = BTreeMap::new();
    for (name, h) in &tracked_inputs {
        input_gradients.insert(name.clone(), grad_of(h, &format!("input {name}"))?);
    }
    Ok(StageBackward {
        loss,
        gradients,
        input_gradients,
        peak_bytes,
    })
}

/// The memory levers on the native engine (ADR-0038): the execution order cut into `segments`
/// slices, as the generic path cuts it, run as stages one after the other in this process. Only the
/// activations at the boundaries are kept between the forward and the backward passes, and each
/// segment holds only its own parameters on the card, which is also what offloading asks for.
fn gradient_cuda_segmented(
    index: usize,
    prepared: &Prepared,
    parameters: BTreeMap<String, TensorF32>,
    inputs: BTreeMap<String, TensorF32>,
    seed: u64,
    segments: usize,
) -> Result<StepResult, ExecError> {
    let n = prepared.report.order.len();
    let bounds = crate::graph::stages::even_bounds(n, segments);
    let stages = crate::graph::stages::stages(&prepared.graph, &prepared.report, &bounds)
        .map_err(|e| shape_err(format!("segments: {e}")))?;
    let device = LibTorchDevice::Cuda(index);
    let mut produced = inputs;
    let (mut peak_activation_bytes, mut peak_parameter_bytes) = (0u64, 0u64);
    for st in &stages {
        let (out, peak) =
            super::stage_forward::<T>(&device, prepared, st, &parameters, &produced, seed)?;
        peak_activation_bytes = peak_activation_bytes.max(peak);
        produced.extend(out);
    }
    let mut received: BTreeMap<String, Vec<TensorF32>> = BTreeMap::new();
    let mut gradients: BTreeMap<String, TensorF32> = BTreeMap::new();
    let mut loss = None;
    for st in stages.iter().rev() {
        let b = stage_backward_cuda(index, prepared, st, &parameters, &produced, &received, seed)?;
        peak_activation_bytes = peak_activation_bytes.max(b.peak_bytes);
        let held: u64 = st
            .params
            .iter()
            .filter_map(|p| parameters.get(p))
            .map(|t| t.data.len() as u64 * 4)
            .sum();
        peak_parameter_bytes = peak_parameter_bytes.max(held);
        if b.loss.is_some() {
            loss = b.loss;
        }
        for (name, g) in b.input_gradients {
            received.entry(name).or_default().push(g);
        }
        // A parameter used by several segments (tied weights) sums their contributions.
        for (name, g) in b.gradients {
            match gradients.get_mut(&name) {
                Some(acc) => acc.data.iter_mut().zip(&g.data).for_each(|(a, b)| *a += *b),
                None => {
                    gradients.insert(name, g);
                }
            }
        }
    }
    // A trained parameter no node uses has a zero gradient, as on the whole-model path.
    for p in prepared.graph.parameters.iter().filter(|p| p.trainable) {
        if !gradients.contains_key(&p.name) {
            let dims: Vec<usize> = p.shape.iter().map(|d| *d as usize).collect();
            gradients.insert(p.name.clone(), zeros_of(&dims)?);
        }
    }
    Ok(StepResult {
        loss: loss.ok_or_else(|| ExecError::Numeric("no stage produced the loss".into()))?,
        gradients,
        peak_activation_bytes,
        peak_parameter_bytes,
    })
}
