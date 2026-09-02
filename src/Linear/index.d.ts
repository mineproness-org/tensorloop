export class Linear{
    Weights : Float32Array[]
    Bias: Float32Array
    constructor(embeddingSize : number, vocabSize : number, configs: {
        save: {
            filename: string[]
        }
    })
    forward(input : Float32Array[]) : Float32Array[]
    backward(outGradient : number[] , learingRate: number) : Float32Array[]
    Save() : void
    ClearInputCache() : void
}