export class SelfAttention{
    q : Float32Array[]
    v : Float32Array[]
    k : Float32Array[]
    input;
    constructor(embeddingSize : number, configs: {
            dirname: string
        
    })
    weights : Float32Array[]
    score : number
    forward(vectors: Float32Array[][])
    Save() : void
    backward(dOutput, learningRate)
}