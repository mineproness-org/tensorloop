import { LoadBias, LoadVectors, SaveBias, SaveVectors } from '../GetConfigs.js'
import { existsSync } from 'fs'
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
    }
    Save() {
        SaveVectors(this.Weights, this.configs.save.filename[0])
        SaveBias(this.Bias, this.configs.save.filename[1])
    }
    forward(input) {
        this.input.push(input)
        const output = new Float32Array(this.Weights.length)
        for (let a = 0; a < this.Weights.length; a++) {
            let sum = this.Bias[a]
            for (let t = 0; t < input.length; t++) {
                sum += input[t] * this.Weights[a][t]
            }
            output[a] = sum
        }
        return output
    }
    backward(outputGradient, learningRate, idx) {
        const input = this.input[idx];
        const weights = this.Weights
        const inputGradient = new Float32Array(input.length)
        for (let a = 0; a < weights.length; a++) {
            const gran = Math.max(-1, Math.min(1, outputGradient[a]))
            const w = weights[a]
            for (let at = 0; at < input.length; at++) {
                inputGradient[at] += gran * w[at]
                w[at] -= learningRate * input[at] * gran
            }
            this.Bias[a] -= learningRate * gran
        }
        return inputGradient
    }
    ClearInputCache(){
        this.input = []
    }
}