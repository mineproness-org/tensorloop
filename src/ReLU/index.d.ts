export class Relu{
    constructor(device : GPUAdapter, embeddingSize : number);
    async forward(input: GPUBuffer) : Promise<GPUBuffer>
    async backward(outGradinet: GPUBuffer) : Promise<GPUBuffer>
}