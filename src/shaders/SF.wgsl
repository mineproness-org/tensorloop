struct Params {
    vocabSize: u32,
    tokenCount: u32,
}

@group(0) @binding(0)
var<storage, read> logits: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> sharedMax: array<f32, 256>;
var<workgroup> sharedSum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,) {
    let token = wg.x;
    let tid = lid.x;
    if (token >= params.tokenCount) {
        return;
    }
    let start = token * params.vocabSize;
    var localMax = - 3.402823e38;
    var i = tid;
    while (i < params.vocabSize) {
        localMax = max(localMax, logits[start + i]);
        i += 256;
    }
    sharedMax[tid] = localMax;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }
    let maxLogit = sharedMax[0];
    var localSum = 0.0;
    i = tid;
    while (i < params.vocabSize) {
        localSum += exp(logits[start + i] - maxLogit);
        i += 256;
    }
    sharedSum[tid] = localSum;
    workgroupBarrier();
    stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            sharedSum[tid] += sharedSum[tid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    let sum = sharedSum[0];
    i = tid;
    while (i < params.vocabSize) {
        output[start + i] = exp(logits[start + i] - maxLogit) / sum;
        i += 256;
    }
}