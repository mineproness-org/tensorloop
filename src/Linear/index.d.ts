export class Linear {
    constructor(device: GPUAdapter, embeddingSize: number, vocabSize: number, configs: {
        cpuReadBack: boolean,
        save: {
            filename: string[]
        }
    })
    async forward(vec: GPUBuffer): Promise<GPUBuffer | Float32Array>
    async backward(dOutput: GPUBuffer, lr: number): Promise<GPUBuffer>
    async Save(): Promise<void>
    async ClearInputCache() : Promise<void>
}