struct Params {
    vocabSize: u32,
    tokenCount: u32,
}

@group(0) @binding(0)

var <storage, read> probs : array<f32>;

@group(0) @binding(1)

var <storage, read> targetToken : array<u32>;

@group(0) @binding(2)

var <storage, read_write> loss : array<f32>;

@group(0) @binding(3)

var <uniform> params: Params;


@compute @workgroup_size(4)

fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let index = id.x;
    var targetIndex = targetToken[index];
    let probalitils = probs[index * params.vocabSize + u32(targetIndex)];
    loss[index] = -log(probalitils + 1e-15);
}