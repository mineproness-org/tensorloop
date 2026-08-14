export class FFN{
    constructor(embeddingSize: number, hiddenLayer: "gelu" | "relu" , configs: {
        dirname: string
    })
    forward(input: Float32Array[]) : Float32Array[]
    backward(dInput: Float32Array[], LearningRate: number) : Float32Array[]
    Save() : void;
    ClearInputCache() : void
}