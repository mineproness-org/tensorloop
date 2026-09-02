import { SelfAttention } from '../SelfAttention/index.js'
import { FFN } from '../FFN/index.js'
import fs from 'fs'
export class TransformerBlock {
    ffn;
    selfAttention;
    constructor(embeddingSize, configs) {
        if(!fs.existsSync(configs.dirname)) fs.mkdirSync(configs.dirname)
        this.ffn = new FFN(embeddingSize, configs.activation, configs)
        this.selfAttention = new SelfAttention(embeddingSize, configs)
    }
    forward(input) {
        const attention = this.selfAttention.forward(input)
        return this.ffn.forward(attention)
    }
    backward(dOutput, learningRate) {
        const dAttention = this.ffn.backward(dOutput, learningRate)
        return this.selfAttention.backward(dAttention, learningRate)
    }
    clearInputCache() {
        this.ffn.ClearInputCache()
        this.selfAttention.key.ClearInputCache()
        this.selfAttention.query.ClearInputCache()
        this.selfAttention.value.ClearInputCache()
    }
    Save() {
        this.selfAttention.Save()
        this.ffn.Save()
    }
}