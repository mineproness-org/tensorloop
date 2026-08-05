export class PositionEmbedding{
    vectors: Float32Array[]
    constructor(embeddingSize: number, contextSize: number, configs: {
        save: {
            filename: string
        }
    })
    forward(position: number | number[]) : Float32Array[] | Float32Array[][]
    backward(pos: number , inputGradient: number[] | Float32Array[], learingRate: number) : void
    Save() : void;
    addPos(PSEM, vectors : Float32Array[][]) : Float32Array 
}