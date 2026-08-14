export class LayerNorm {
    gamma: Float32Array;
    beta: Float32Array;
    eps: number;
    cache: Array<{
        input: Float32Array;
        mean: number;
        invStd: number;
    }>;

    constructor(embeddingSize: number, eps?: number);

    forward(input: Float32Array, index?: number): Float32Array;

    backward(
        dOutput: Float32Array,
        learningRate: number,
        index?: number
    ): Float32Array;

    ClearInputCache(): void;
}