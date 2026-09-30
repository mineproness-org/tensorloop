struct Params {
    vocabSize: u32,
    tokenCount: u32,
}

@group(0) @binding(0)

var <storage, read> probs : vec2<f32>;

@group(0) @binding(1)

var <storage, read> targetToken : array<f32>;

@group(0) @binding(2)

var <storage, read_write> loss : vec2<f32>;

@group(0) @binding(3)

var <uniform> params: Params;


@compute @workgroup_size(256)

fn main(@builtin(global_invocation_id) id : vec3<u32>) {
      let index = id.x;
    let token = index / params.vocabSize;
    let vocabIndex = index % params.vocabSize;
    let start = token * params.vocabSize;
    let end = start + params.vocabSize;
    var targetIndex = targetToken[token];
    let total: u32 = params.tokenCount * params.vocabSize;
    let probalitils = probs[u32(targetIndex)];
    loss[index] = -log(probalitils + 1e-15);
}