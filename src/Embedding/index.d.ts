export class Embedding{
    vectors: Float32Array[]
    constructor(embeddingSize: number, vocabSize: number, configs: {
        save: {
            filename: string
        }
    })
    forward(token: number | number[]) : Float32Array[] | Float32Array[][]
    backward(token: number , inputGradient: number[] | Float32Array[], learingRate: number) : void
}