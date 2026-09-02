import { LoadBias, LoadVectors, SaveBias, SaveVectors } from '../GetConfigs.js'
import { existsSync } from 'fs'
import { GPULinear } from '../gpiInit.js'
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
    gpu;
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
        const {bias, weights} = this.gpu.Save()
        SaveVectors(weights, this.configs.save.filename[0])
        SaveBias(bias, this.configs.save.filename[1])
    }
    forward(input) {
        if (!this.gpu) {
            this.gpu = new GPULinear(this.Weights, this.Bias, input.length)
        }
        for (let a = 0; a < input.length; a++) {
            this.input[a] = input;
        }
        return this.gpu.forward(input)
    }
    backward(out, lr){
        return this.gpu.backward(out, this.input, lr)
    }
    ClearInputCache() {
        this.input = []
    }
}


