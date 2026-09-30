export class TransformerBlock{
    constructor(device: GPUAdapter, embeddingSize: number, configs: {
        dirname: string
    })
    async forward(input : GPUBuffer) : Promise<GPUBuffer>
    async backward(outInput : GPUBuffer, lr : number) : Promise<GPUBuffer>
    async ClearInputCache() : Promise<void>
    async Save() : Promise<void>
}