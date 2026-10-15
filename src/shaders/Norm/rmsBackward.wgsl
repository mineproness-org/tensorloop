
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2)
var<storage, read> sumSquares: array<f32>;

@group(0) @binding(3)
var<storage, read> gamma: array<f32>;

@group(0) @binding(4)
var<storage, read_write> gradInput: array<f32>;

var<workgroup> partial: array<f32, 64>;

const D: u32 = %embeddingSize%u;
const EPSILON: f32 = 1e-6;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let token = wid.x;
    if (token >= arrayLength(&sumSquares)) {
        return;
    }
    let base = token * D;
    let r = sqrt(sumSquares[token] / f32(D) + EPSILON);
    var localSum = 0.0;
    var d = lid;
    while (d < D) {
        let idx = base + d;
        localSum += gradOutput[idx] * gamma[d] * input[idx];
        d += 64u;
    }

    partial[lid] = localSum;
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
    let dot = partial[0];
    d = lid;
    while (d < D) {
        let idx = base + d;
        let x = input[idx];
        let dy = gradOutput[idx];
        let g = gamma[d];

        gradInput[idx] =
            (dy * g) / r
            - (x * dot) / (f32(D) * r * r * r);

        d += 64u;
    }
}