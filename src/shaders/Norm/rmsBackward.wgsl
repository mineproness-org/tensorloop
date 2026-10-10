
@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> gradOutput: array<f32>;

@group(0) @binding(2)
var<storage, read> rms: array<f32>;

@group(0) @binding(3)
var<storage, read_write> gradInput: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32) {
    let D = arrayLength(&input);
    partial[lid] = gradOutput[lid] * input[lid];
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
    let dot = partial[0];
    let r = rms[0];
    gradInput[lid] = gradOutput[lid] / r - input[lid] * dot / (f32(D) * r * r * r);
}