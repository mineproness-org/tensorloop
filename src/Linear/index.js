import { GetVector, GetVectors, Save, SaveVectors } from '../GetConfigs.js'
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
                const Vectors = GetVectors(configs.save.filename[0], embeddingSize)
                const Bias = GetVector(configs.save.filename[1])
                this.Weights = Vectors;
                this.Bias = Bias
            } else {
                const { vectors, Bias } = GenerateWeightsBias(embeddingSize, vocabSize)
                this.Weights = vectors;
                this.Bias = Bias;
                SaveVectors(configs.save.filename[0], this.Weights)
                Save(configs.save.filename[1], this.Bias)
            }
        } else {
            const { vectors, Bias } = GenerateWeightsBias(embeddingSize, vocabSize)
            this.Weights = vectors;
            this.Bias = Bias;
        }
    }
    Save() {
        SaveVectors(this.configs.save.filename[0], this.Weights)
        Save(this.configs.save.filename[1], this.Bias)
    }
    forward(input) {
        this.input = input
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
    backward(outputGradient, learningRate) {
        const inputGradient = new Float32Array(this.input.length)
        for (let a = 0; a < this.Weights.length; a++) {
            for (let at = 0; at < this.input.length; at++) {
                inputGradient[at] += outputGradient[a] * this.Weights[a][at]
                this.Weights[a][at] -= learningRate * this.input[at] * outputGradient[a]
            }
            this.Bias[a] -= learningRate * outputGradient[a]
        }
        return inputGradient
    }
}