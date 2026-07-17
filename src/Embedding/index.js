import fs, { existsSync } from 'fs'
import { GetVectors, SaveVectors } from '../GetConfigs.js'
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
    constructor(embeddingSize, vocabSize, configs) {
        this.configs = configs
        if (configs && configs.save) {
            if (existsSync(configs.save.filename)) {
                 this.vectors = GetVectors(configs.save.filename, embeddingSize)
            } else {
                this.vectors = GenerateVectors(embeddingSize, vocabSize)
                SaveVectors(configs.save.filename, this.vectors)
            }
        } else {
            this.vectors = GenerateVectors(embeddingSize, vocabSize)

        }
    }
    forward(token) {
       if(Array.isArray(token)){
        return token.map((e)=> this.vectors[e])
       }else{
        return this.vectors[token]
       }
    }
    backward(token, inputGradient, learingRate){
        for(let a = 0; a < this.vectors[token].length; a++){
           this.vectors[token][a] -= learingRate * inputGradient[a]
        }
    }
    Save(){
        SaveVectors(this.configs.save.filename , this.vectors)
    }
}