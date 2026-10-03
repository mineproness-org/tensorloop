struct Params {
    vocabSize: f32,
    tokenCount: f32,
    lr: f32,
    inputSize: f32
}

@group(0) @binding(0)

var<storage, read_write> bias: array<f32>;

@group(0) @binding(1)

var<storage, read_write> weight: array<f32>;

@group(0) @binding(2)

var<storage, read_write> outGradient: array<f32>;

@group(0) @binding(3)

var<storage, read_write> input: array<f32>;

@group(0) @binding(4)

var<uniform> params: Params;

@compute @workgroup_size(256)

fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    let totalWeights = u32(params.inputSize * params.vocabSize);
    let token = index / u32(params.inputSize);
    if (index >= totalWeights) {
        return;
    }
    let outputIndex = index / u32(params.inputSize);
    let inputIndex = index % u32(params.inputSize);
    var dWeights = 0.0;
    for (var token = 0u; token < u32(params.tokenCount); token++) {
        let inputValue = input[token * u32(params.inputSize) + inputIndex];
        let gradient = outGradient[token * u32(params.vocabSize) + outputIndex];
        let dW = inputValue * gradient;
        dWeights += f32(dW);
    }
    let vClap = weight[index] - params.lr * max(- 7.0, min(7.0, dWeights));
    weight[index] = vClap;
    if (inputIndex == 0u) {
        var dBias = 0.0;
        for (var token = 0u; token < u32(params.tokenCount); token++) {
            dBias += f32(outGradient[token * u32(params.vocabSize) + outputIndex]);
        }
        bias[outputIndex] = bias[outputIndex] - params.lr * max(- 7.0, min(7.0, dBias));

    }
}