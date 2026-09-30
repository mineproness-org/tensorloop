struct Params{
    vocabSize: u32,
    tokenCount: u32
}

@group(0) @binding(0)

var <uniform> params : Params;


@group(0) @binding(1)

var <storage, read_write> probs : array<f32>;

@group(0) @binding(3)

var <storage, read_write> tToken : array<u32>;

@group(0) @binding(2)


var <storage, read_write> result : array<f32>;


@compute @workgroup_size(1)

fn main(@builtin(global_invocation_id) id : vec3<u32>){
    let x = id.x;

    var correct = 0.0;
    for(var a = 0u; a < params.tokenCount; a++){
        var bestVal = - 3.402823e38;
        var maxToken = 0u;
        var start = a * params.vocabSize;
        let end = start + params.vocabSize;
        for(var t = start; t < end; t++){
            let val = probs[t];
            if(bestVal < val){
                bestVal = val;
                maxToken = t - start;
            }
        }
        if(maxToken == tToken[a]){
            correct += 1.0;
        }

    }
    correct /= f32(params.tokenCount);
    result[x] = correct;
}