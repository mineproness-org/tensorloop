export class TransformerBlock{
    constructor(embeddingSize: number, configs:{
        dirname: string,
        activation: "relu" | "gelu"
    })
    Save(): void
    clearInputCache() : void
    forward(input: Float32Array[]) : Float32Array[]
    backward(dOutput: Float32Array[], LearningRate: number) : Float32Array[]
}