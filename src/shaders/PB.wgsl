struct Params{
    lr : f32
}

@group(0) @binding(0)
var <storage, read_write> posVec: array<f32>;

@group(0) @binding(1)

var <storage, read> outputGradient : array<f32>;


@group(0) @binding(2)

var <uniform> params: Params;

@compute @workgroup_size(64)

fn main(@builtin(global_invocation_id) id : vec3<u32>){
    let x = id.x;
    if(x >= arrayLength(&outputGradient)){
        return;
    }
    posVec[x] -= params.lr * outputGradient[x];
}