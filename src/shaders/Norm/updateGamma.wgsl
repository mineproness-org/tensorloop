
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> gradOutput: array<f32>;

@group(0) @binding(2)
var<storage, read> sumSquares: array<f32>;

@group(0) @binding(3)
var<storage, read_write> gamma: array<f32>;

@group(0) @binding(4)
var<storage, read> params: array<f32>;

var<workgroup> partial: array<f32, 64>;

const D: u32 = %embeddingSize%u;
const EPSILON: f32 = 1e-6;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let dim = wid.x * 64u + lid;
    let T = arrayLength(&sumSquares);
    var sum = 0.0;
    if (dim < D) {
        var token = lid;

        while (token < T) {
            let idx = token * D + dim;
            let r = sqrt(sumSquares[token] / f32(D) + EPSILON);
            sum += gradOutput[idx] * input[idx] / r;
            token += 64u;
        }
    }
    partial[lid] = sum;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    if (lid == 0u && wid.x * 64u < D) {
        let dimToUpdate = wid.x * 64u;
        gamma[dimToUpdate] -= params[0] * partial[0];
    }
}