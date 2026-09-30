export class Embedding{
    constructor(device: GPUBuffer ,embedding : number, vocabSize: number, config:{
        cpuReadBack: boolean,
        save: {
            filename: string
        }
    })
    async forward(tokens: number[]) : Promise<GPUBuffer>
    async Save() : void
    async backward(tokens: number[], inputGradient: GPUBuffer, lr: number) : Promise<void>
}