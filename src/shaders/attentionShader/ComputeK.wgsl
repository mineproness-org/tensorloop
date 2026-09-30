struct Params {
    contextSize: u32,
    embeddingSize: u32,
}
@group(0) @binding(0)
var<storage, read> dS: array<f32>;
@group(0) @binding(1)
var<storage, read> Q: array<f32>;
@group(0) @binding(2)
var<storage, read_write> dK: array<f32>;
@group(0) @binding(3)
var<uniform> params: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    let T = params.contextSize;
    let D = params.embeddingSize;
    if (row >= T || col >= D) {
        return;
    }
    var sum = 0.0;
    for (var i = 0u; i < T; i++) {
        let dsIndex = i * T + row;
        let qIndex = i * D + col;
        sum += dS[dsIndex] * Q[qIndex];
    }
    let scale = sqrt(f32(D));
    dK[row * D + col] = sum / scale;
}