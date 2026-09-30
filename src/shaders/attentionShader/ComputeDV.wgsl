struct Params {
    contextSize: u32,
    embeddingSize: u32,
}

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> dOutput: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dV: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;


@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>)
 {
    let row = id.x;
    let col = id.y;
    let T = params.contextSize;
    let D = params.embeddingSize;
    if (row >= T || col >= D) {
        return;
    }
    var sum = 0.0;
     for (var i = 0u; i < T; i++) {
        let aIndex = i * T + row;
        let gradIndex = i * D + col;
        sum += A[aIndex] * dOutput[gradIndex];
    }
    let outputIndex = row * D + col;
    dV[outputIndex] = sum;
}