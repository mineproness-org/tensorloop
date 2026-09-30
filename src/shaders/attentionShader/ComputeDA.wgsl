struct Params {
    contextSize: u32,
    embeddingSize: u32,
}
@group(0) @binding(0)
var<storage, read> dOutput: array<f32>;
@group(0) @binding(1)
var<storage, read> V: array<f32>;
@group(0) @binding(2)
var<storage, read_write> dA: array<f32>;
@group(0) @binding(3)
var<uniform> params: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    let T = params.contextSize;
    let D = params.embeddingSize;
    if (row >= T || col >= T) {
        return;
    }
    var sum = 0.0;
    for (var k = 0u; k < D; k++) {
        let dOutputIndex = row * D + k;
        let vIndex = col * D + k;
        sum += dOutput[dOutputIndex] * V[vIndex];
    }
    let outputIndex = row * T + col;
    dA[outputIndex] = sum;
}