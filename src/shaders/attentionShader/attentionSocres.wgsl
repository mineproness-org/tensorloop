struct Params {
    contextSize: u32,
    embeddingSize: u32,
}

@group(0) @binding(0)
var<storage, read> Q: array<f32>;

@group(0) @binding(1)
var<storage, read> K: array<f32>;

@group(0) @binding(2)
var<storage, read_write> scores: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;

    if (row >= params.contextSize || col >= params.contextSize) {
        return;
    }
    var sum = 0.0;
    for (var k = 0u; k < params.embeddingSize; k++) {
        let qIndex = row * params.embeddingSize + k;
        let kIndex = col * params.embeddingSize + k;
        sum += Q[qIndex] * K[kIndex];
    }

    let outIndex = row * params.contextSize + col;
    if (col > row) {
        scores[outIndex] = - 1e9;
        return;
    }
    let scale = sqrt(f32(params.embeddingSize));
    scores[outIndex] = sum / scale;
}