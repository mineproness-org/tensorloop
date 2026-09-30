export class FFN{
    constructor(device: GPUAdapter, embeddingSize: number, configs: {
        dirname: string
    })
    async forward(vec: GPUBuffer) : Promise<GPUBuffer>
    async backward(inputGradient: GPUBuffer, lr: number) : Promise<GPUBuffer>
      async Save() : Promise<void>
      async ClearInputCache() : Promise<void>
}