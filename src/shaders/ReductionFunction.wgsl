struct Parmas {
    vocabSize: u32,
    count: u32
}

@group(0) @binding(0)

var<storage, read> data: array<f32>;

@group(0) @binding(1)

var<storage, read_write> result: array<f32>;

@group(0) @binding(2)

var<uniform> params: Parmas;

@compute @workgroup_size(1)

fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    var sum = 0.0;
    for (var i = 0u; i < params.count; i++) {
        sum += data[i];
    }
    sum /= f32(params.count);
    result[index] = sum;
}