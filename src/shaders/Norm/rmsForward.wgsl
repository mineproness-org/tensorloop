@group(0) @binding(0)
var<storage, read> reduce: array<f32>;

@group(0) @binding(1)
var<storage, read> vector: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<storage, read> gamma: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;

    if (x >= arrayLength(&vector)) {
        return;
    }

    let dim = %embeddingSize%u;
    let token = x / dim;
    let d = x % dim;

    let rms = sqrt(
        reduce[token] / f32(dim) + 1e-6
    );

    output[x] = (vector[x] / rms) * gamma[d];
}