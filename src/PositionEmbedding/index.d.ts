export class PositionEmbedding{
    constructor(device: GPUBuffer ,embedding : number, contextSize: number, config:{
        save: {
            filename: string
        }
    });
    async forward(input: GPUBuffer) : Promise<GPUBuffer>
    async Save() : Promise<GPUBuffer>
    async backward(inputGradient: GPUBuffer, lr: number) : Promise<void>
}