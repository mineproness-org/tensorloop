export class RMSNorm{
    constructor(device : GPUAdapter, embeddingSize : number);
    async forward(input : GPUBuffer) : Promise<GPUBuffer>
    async backward(outputGradient : GPUBuffer) : Promise<GPUBuffer>
}