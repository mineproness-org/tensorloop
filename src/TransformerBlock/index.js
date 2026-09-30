import {SelfAttention} from '../SelfAttention/index.js'
import {FFN} from '../FFN/index.js'
import {existsSync, mkdirSync} from 'fs'

export class TransformerBlock{
    attention;
    ffn;
    constructor(device, embeddingSize, configs){
        if(!existsSync(configs.dirname)) mkdirSync(configs.dirname);
        this.attention = new SelfAttention(device, embeddingSize, configs)
        this.ffn = new FFN(device, embeddingSize, configs)
    }
    async forward(input){
        const xAtt = await this.attention.forward(input);
        const xFF = await this.ffn.forward(xAtt);
        return xFF
    }
    async backward(outGradient, lr){
        const dAtt = await this.ffn.backward(outGradient, lr);
        const dFF = await   this.attention.backward(dAtt, lr)
        return dFF
    }
    async ClearInputCache(){
        await this.ffn.ClearInputCache()
        await this.attention.ClearInputCache()
    }
    async Save(){
        await this.ffn.Save()
        await this.attention.Save()
    }
}