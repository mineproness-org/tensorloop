import { LoadBias, LoadVectors, SaveBias, SaveVectors } from '../GetConfigs.js'
import { existsSync } from 'fs'
import { CalculateBackward, CalculateLinearForward, InitKernel } from '../GPU.js'
function GenerateWeightsBias(embeddingSize, vocabSize) {
    const vectors = []
    const Bias = new Float32Array(vocabSize)
    for (let a = 0; a < vocabSize; a++) {
        vectors[a] = new Float32Array(embeddingSize)
        for (let t = 0; t < embeddingSize; t++) {
            vectors[a][t] = (Math.random() * 2 - 1) * 0.02
        }
        Bias[a] = 0
    }
    return { vectors, Bias }
}

export class Linear {
    Weights = []
    Bias = []
    configs;
    input = [];
    constructor(embeddingSize, vocabSize, configs) {
        if (configs && configs.save) {
            this.configs = configs
            if (existsSync(configs.save.filename[0]) && existsSync(configs.save.filename[1])) {
                const Vectors = LoadVectors(configs.save.filename[0], embeddingSize)
                const Bias = LoadBias(configs.save.filename[1])
                this.Weights = Vectors;
                this.Bias = Bias
            } else {
                const { vectors, Bias } = GenerateWeightsBias(embeddingSize, vocabSize)
                this.Weights = vectors;
                this.Bias = Bias;
                SaveVectors(this.Weights, configs.save.filename[0])
                SaveBias(this.Bias, configs.save.filename[1])
            }
        } else {
            const { vectors, Bias } = GenerateWeightsBias(embeddingSize, vocabSize)
            this.Weights = vectors;
            this.Bias = Bias;
        }
        InitKernel(this.Weights, embeddingSize)
    }
    Save() {
        SaveVectors(this.Weights, this.configs.save.filename[0])
        SaveBias(this.Bias, this.configs.save.filename[1])
    }
    forward(input) {
        this.input.push(input)
        return CalculateLinearForward(this.Weights, this.Bias, input)
    }
    backward(outputGradient, learningRate, idx) {
        const input = this.input[idx];
        const weights = this.Weights
        const {biasOut,inputGradient, weightsOut} = CalculateBackward(weights, this.Bias, outputGradient, learningRate, input)
        this.Weights = weightsOut;
        this.Bias = biasOut
        return inputGradient
    }
    ClearInputCache(){
        this.input = []
    }
}