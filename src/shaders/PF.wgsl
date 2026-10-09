@group(0) @binding(0)
var <storage, read> posVec: array<f32>;

@group(0) @binding(1)

var <storage, read> vecs : array<f32>;

@group(0) @binding(2)

var <storage, read_write> output: array<f32>;


@compute @workgroup_size(64)

fn main(@builtin(global_invocation_id) id : vec3<u32>){
    let x = id.x;
    if(x >= arrayLength(&vecs)){
        return;
    }
    output[x] = vecs[x] + posVec[x];
}