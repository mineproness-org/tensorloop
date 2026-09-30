struct Params {
    tokens: u32,
    embeddingSize: u32,
}
@group(0) @binding(0)
var<storage, read> weights: array<f32>;
@group(0) @binding(1)
var<storage, read> value: array<f32>;
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;
@group(0) @binding(3)
var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total =params.tokens * params.embeddingSize;
    if (index >= total) {
        return;
    }
    let row = index / params.embeddingSize;
    let dim = index % params.embeddingSize;
    var sum = 0.0;
    for (var j = 0u; j < params.tokens; j++) {
        let weight = weights[row * params.tokens + j];
        let v =value[j * params.embeddingSize + dim];
        sum += weight * v;
    }
    output[index] = sum;
}