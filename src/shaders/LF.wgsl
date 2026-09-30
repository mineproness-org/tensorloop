 struct Params {
    inputSize: u32,
    outputSize: u32,
    tokenCount: u32,
}

@group(0) @binding(0)

var<storage, read_write> weights: array<f32>;

@group(0) @binding(1)

var<storage, read_write> output: array<f32>;

@group(0) @binding(2)

var<storage, read_write> input: array<f32>;

@group(0) @binding(3)

var<storage, read_write> bias: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

@compute @workgroup_size(256)

fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total = params.tokenCount * params.outputSize;
    if (index >= total) {
        // output[index] = f32(total);
        return;
    }
    let token = index / params.outputSize;
    let outIndex = index % params.outputSize;
    var sum = bias[outIndex];
    for (var i = 0u; i < params.inputSize; i++) {
        let inputIndex = token * params.inputSize + i;
        let weightIndex = outIndex * params.inputSize + i;
        sum += input[inputIndex] * weights[weightIndex];
    }
    output[index] = sum;
}