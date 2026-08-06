import fs, { existsSync } from 'fs'
import { LoadVectors, SaveVectors } from '../GetConfigs.js'
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

export class PositionEmbedding {
    vectors = []
    configs;
    embeddingSize = 0
    constructor(embeddingSize, vocabSize, configs) {
        this.configs = configs
        this.embeddingSize = embeddingSize
        if (configs && configs.save) {
            if (existsSync(configs.save.filename)) {
                this.vectors = LoadVectors(configs.save.filename, embeddingSize)
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
                return this.vectors[e]
            })
        } else {

            return this.vectors[token]
        }
    }
    backward(token, inputGradient, learingRate) {
        for (let a = 0; a < this.vectors[token].length; a++) {
            this.vectors[token][a] -= learingRate * inputGradient[a]
        }
    }
    Save() {
        SaveVectors(this.vectors, this.configs.save.filename)
    }
    addPos(xEM) {
        const out = []

        for (let pos = 0; pos < xEM.length; pos++) {
            const tokenVector = xEM[pos]
            const posVector = this.forward(pos)
            const x = new Float32Array(xEM[0].length)
            for (let a = 0; a < tokenVector.length; a++) {
                x[a] = tokenVector[a] + posVector[a]
            }
            out.push(x)
        }
        return out
    }
}