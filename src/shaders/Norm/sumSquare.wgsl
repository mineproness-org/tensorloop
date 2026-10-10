
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32) {
    var sum = 0.0;
    var i = lid;
    while (i < arrayLength(&input)) {
        sum += input[i] * input[i];
        i += 64u;
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
        stride = stride / 2u;
    }
    if (lid == 0u) {
        out[0] = partial[0];
    }
}