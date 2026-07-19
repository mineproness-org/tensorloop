import fs, { existsSync } from 'fs'
import { GetEmbeddingVectors, SaveEmbeddingVectors, SaveVectors } from '../GetConfigs.js'
// import { config } from 'process'

function GenerateVectors(embeddingSize, vocabSize) {
    const vector = []
    for (let token = 0; token < vocabSize; token++) {
        vector[token] = new Float32Array(embeddingSize)
        for (let a = 0; a < embeddingSize; a++) {
            vector[token][a] = (Math.random() * 2 - 1) * 0.2
        }
    }
    return vector
}

export class Embedding {
    vectors = []
    configs;
    embeddingSize = 0
    constructor(embeddingSize, vocabSize, configs) {
        this.configs = configs
        this.embeddingSize = embeddingSize
        if (configs && configs.save) {
            if (existsSync(configs.save.filename)) {
                this.vectors = []
            } else {
                this.vectors = GenerateVectors(embeddingSize, vocabSize)
                this.Save()
            }
        } else {
            this.vectors = GenerateVectors(embeddingSize, vocabSize)

        }
    }
    forward(token) {
        if (Array.isArray(token)) {
            return token.map((e) => {
                const id = this.vectors[e]
                if (id) return id
                this.vectors[e] = GetEmbeddingVectors(e, this.embeddingSize, this.configs.save.filename)
                return this.vectors[e]
            })
        } else {
            if (!this.vectors[token]) {
                this.vectors[token] = GetEmbeddingVectors(token, this.embeddingSize, this.configs.save.filename)
            }
            return this.vectors[token]
        }
    }
    backward(token, inputGradient, learingRate) {
        for (let a = 0; a < this.vectors[token].length; a++) {
            this.vectors[token][a] -= learingRate * inputGradient[a]
        }
    }
    Save() {
        SaveEmbeddingVectors(this.vectors, this.configs.save.filename)
    }
}