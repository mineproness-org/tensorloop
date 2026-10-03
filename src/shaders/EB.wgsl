struct Params {
    lr: f32
}

@group(0) @binding(0)

var<storage, read_write> embeddings: array<f32>;

@group(0) @binding(1)

var<storage, read_write> outputGradient: array<f32>;

@group(0) @binding(2)

var<storage, read_write> tokens: array<u32>;

@group(0) @binding(3)
var<uniform> params: Params;

const embeddingSize = % embeddingSize % u;

@compute @workgroup_size(256)

fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = id.x % embeddingSize;
    if (id.x >= arrayLength(&outputGradient)) {
        return;
    }
    let tokenIndex = id.x / embeddingSize;
    let tokenID = tokens[tokenIndex];
    let embeddingIDx = tokenID * embeddingSize + dim;
    embeddings[embeddingIDx] -= params.lr * outputGradient[id.x];
}