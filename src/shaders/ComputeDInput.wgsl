struct Params {
    vocabSize: f32,
    tokenCount: f32,
    lr: f32,
    inputSize: f32,
}

@group(0) @binding(0)
var<storage, read_write> weight: array<f32>;
@group(0) @binding(1)
var<storage, read> outGradient: array<f32>;
@group(0) @binding(3)
var<uniform> params: Params;
@group(0) @binding(4)
var<storage, read_write> dInput: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total = params.tokenCount * params.inputSize;
    if (index >= u32(total)) {
        return;
    }
    let token =index / u32(params.inputSize);
    let inputIndex = index % u32(params.inputSize);
    var gradInput = 0.0;
    for (var i = 0u; i < u32(params.vocabSize); i++) {
        let w = weight[i * u32(params.inputSize) + inputIndex];
        let grad = outGradient[token * u32(params.vocabSize) + i];
        gradInput += grad * w;
    }
    dInput[index] = max(-7.0, min(7.0,gradInput));
}