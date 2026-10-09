import {SelfAttention} from '../SelfAttention/index.js'
import {FFN} from '../FFN/index.js'
import {existsSync, mkdirSync} from 'fs'
import {OperationManager} from '../OperationManager.js'
export class TransformerBlock{
    attention;
    ffn;
    gpu;
    size;
    constructor(device, embeddingSize, configs){
        this.size = embeddingSize;
        if(!existsSync(configs.dirname)) mkdirSync(configs.dirname);
        this.attention = new SelfAttention(device, embeddingSize, configs)
        this.ffn = new FFN(device, embeddingSize, configs)
        this.gpu = new OperationManager(device)
    }
    async forward(input){
        let context = input.size / (this.size * 4)
        const xAtt = await this.attention.forward(input);
        const re1 = await this.gpu.add(input, xAtt, this.size * context)
        const xFF = await this.ffn.forward(re1);
        const re2 = await this.gpu.add(re1, xFF, this.size * context)
        xAtt.destroy()
        xFF.destroy()
        return re2
    }
    async backward(outGradient, lr){
        const context = outGradient.size / (this.size * 4)
        const dAtt = await this.ffn.backward(outGradient, lr);
        const re1 = await this.gpu.add(outGradient, dAtt, this.size * context)
        const dFF = await this.attention.backward(re1, lr)
        const re2 = await this.gpu.add(re1, dFF , context)
        outGradient.destroy()
        re1.destroy()
        dFF.destroy()
        dAtt.destroy()
        return re2
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