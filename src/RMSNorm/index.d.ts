export class RMSNorm{
    constructor(device : GPUAdapter, embeddingSize : number , {
        save:{
            filename: string
        }
    });
    async forward(input : GPUBuffer) : Promise<GPUBuffer>
    async backward(outputGradient : GPUBuffer) : Promise<GPUBuffer>
    async Save() : Promise<void>
}