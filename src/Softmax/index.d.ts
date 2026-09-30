export class Softmax{
    constructor(device: GPUAdapter, embeddingSize: number, vocabSize: number);
    async forward(logit: GPUBuffer, isCpuReadBack) : Promise<Float32Array | GPUBuffer>;
    async backward(probs : GPUBuffer, target : number[]) : Promise<GPUBuffer>
    async crossEntropy(probs: GPUBuffer, target: number[]) : Promise<{loss: Float32Array, acc: Float32Array}>
}