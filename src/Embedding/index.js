import fs, { existsSync } from 'fs'
import { SaveVectors, LoadVectors } from '../GetConfigs.js'
import { GPUEmbedding } from '../gpiInit.js'
// import { GPUEmbedding } from '../GPU.js'
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
    gpu;
    constructor(embeddingSize, vocabSize, contextSize, configs) {
        this.configs = configs
        this.embeddingSize = embeddingSize
        if (configs && configs.save) {
            if (existsSync(configs.save.filename)) {
                this.vectors = LoadVectors(configs.save.filename, embeddingSize)
            } else {
                this.vectors = GenerateVectors(embeddingSize, vocabSize)
                SaveVectors(this.vectors, this.configs.save.filename)
            }
        } else {
            this.vectors = GenerateVectors(embeddingSize, vocabSize)

        }  
        this.gpu = new GPUEmbedding(this.vectors, vocabSize, embeddingSize, contextSize)
    }
    forward(token) {
        return this.gpu.forward(token)
    }
    backward(inputGradient, learingRate){
         this.gpu.backward(inputGradient, learingRate)
    }
    Save() {
        const vectors = []
        for(let i = 0; i < this.gpu.vectors.length; i++){
            const out = this.gpu.vectors[i].toArray()
            vectors.push(...out)
        }
        SaveVectors(vectors.map((e)=> new Float32Array(e)), this.configs.save.filename)
    }
}