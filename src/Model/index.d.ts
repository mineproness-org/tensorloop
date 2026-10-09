export class Model {
    async forward(input: Number[], buffer: boolean): Promise<Float32Array | GPUBuffer>;
    async backward(output: GPUBuffer, target: number[], lr: number): Promise<{ acc: number, loss: number }>
    TrainTokenizer(dataset: string): void;
    constructor(device :GPUAdapter, embeddingSize: number, configs: {
        tokenizerName: string
        blockSize: number,
        dirname: string,
        contextSize: number,
        gpuName: string

    })
    async Save() : Promise<void>
    async CreateContext()
}