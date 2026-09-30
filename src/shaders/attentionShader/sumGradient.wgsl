struct Params {
    contextSize: u32,
    embeddingSize: u32,
}

@group(0) @binding(0)
var<storage, read> dXq: array<f32>;
@group(0) @binding(1)
var<storage, read> dXk: array<f32>;
@group(0) @binding(2)
var<storage, read> dXv: array<f32>;
@group(0) @binding(3)
var<storage, read_write> dX: array<f32>;
@group(0) @binding(4)
var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total = params.contextSize * params.embeddingSize;
    if (index >= total) {
        return;
    }
    dX[index] =dXq[index] +dXk[index] +dXv[index];
}