@group(0) @binding(0)

var <storage, read> reduce: array<f32>;

@group(0) @binding(1)

var <storage, read> vector: array<f32>;

@group(0) @binding(2)

var <storage, read_write> output : array<f32>;

@compute @workgroup_size(64)

fn main(@builtin(global_invocation_id) id : vec3<u32>){
    let x = id.x;
    let rms = sqrt(reduce[0u] / f32(arrayLength(&vector)) + 1e-6);
    output[x] = vector[x] / rms;
}