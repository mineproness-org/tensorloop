export class SelfAttention{
    constructor(device : GPUAdapter,embeddingSize: number, configs : {
        dirname: string
    })
    async forward(vec : GPUBuffer) : Promise<GPUBuffer>
    async backward(dO : GPUBuffer, lr : number) : Promise<GPUBuffer>
    async ClearInputCache() : Promise<void>
    async Save() : Promise<void>
} 