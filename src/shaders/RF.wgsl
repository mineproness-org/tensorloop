struct Param {
    embeddingSize: u32,
    tokenCount: u32
}

@group(0) @binding(0)

var<storage, read> input: vec2<f32>;

@group(0) @binding(1)

var<uniform> params: Param;

@group(0) @binding(2)

var<storage, read_write> output: vec2<f32>;

@compute @workgroup_size(256)

fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total = params.embeddingSize * params.tokenCount;
    if (index >= total) {
        return;
    }
    let x = input[index];
    if (x > 0) {
        output[index] = x;
    }
    else {
        output[index] = x * 0.02;
    }
}