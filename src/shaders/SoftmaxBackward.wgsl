struct Params {
    tokens: u32,
    embeddingSize: u32,
}
@group(0) @binding(0)
var<storage, read> P: array<f32>;
@group(0) @binding(1)
var<storage, read> dP: array<f32>;
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;
@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(256)

fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total = params.tokens * params.tokens;
    if (index >= total) {
        return;
    }
    let row = index / params.tokens;
    var dot = 0.0;
    for (var i = 0u; i < params.tokens; i++) {
        let pos = row * params.tokens + i;
        dot += P[pos] * dP[pos];
    }
    let p = P[index];
    let dp = dP[index];
    output[index] = p * (dp - dot);
}