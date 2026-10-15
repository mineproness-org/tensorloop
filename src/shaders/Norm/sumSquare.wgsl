
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 64>;
const D: u32 = %embeddingSize%u;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let base = wid.x * D;
    var sum = 0.0;
    var d = lid;
    while (d < D) {
        let idx = base + d;
        if (idx < arrayLength(&input)) {
            sum += input[idx] * input[idx];
        }
        d += 64u;
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
    if (lid == 0u) {
        out[wid.x] = partial[0];
    }
}