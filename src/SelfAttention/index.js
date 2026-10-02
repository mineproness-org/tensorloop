import { Linear } from "../../index.js";
import { join } from 'path'
import { OperationManager } from "../OperationManager.js";

export class SelfAttention {
    query;
    value;
    key;
    q;
    k;
    v;
    embeddingSize;
    params;
    attentionscoresShader;
    softmaxForward;
    gpu;
    input;
    constructor(device, embeddingSize, configs) {
        this.embeddingSize = embeddingSize;
        const dd = [1, 2, 3].map((e) => {
            return {
                save: {
                    filename: [join(configs.dirname, `SELF-W-${e}.bin`), join(configs.dirname, `SELF-B-${e}.bin`)]
                }
            }
        })
        this.q = new Linear(device, embeddingSize, embeddingSize, dd[0])
        this.k = new Linear(device, embeddingSize, embeddingSize, dd[1])
        this.v = new Linear(device, embeddingSize, embeddingSize, dd[2])
        this.gpu = new OperationManager(device)
        this.attentionscoresShader = this.gpu.getShaderPipline("./src/shaders/attentionShader/attentionSocres.wgsl")
        this.softmaxForward = this.gpu.getShaderPipline("./src/shaders/SF.wgsl")
        this.softmaxBackward = this.gpu.getShaderPipline("./src/shaders/SoftmaxBackward.wgsl")
        this.outputForward = this.gpu.getShaderPipline("./src/shaders/attentionShader/CompueFinalForwardOutput.wgsl")
        this.dVShader = this.gpu.getShaderPipline("./src/shaders/attentionShader/ComputeDV.wgsl")
        this.dAshader = this.gpu.getShaderPipline("./src/shaders/attentionShader/ComputeDA.wgsl")
        this.dQshader = this.gpu.getShaderPipline("./src/shaders/attentionShader/ComputedQ.wgsl")
        this.dKShader = this.gpu.getShaderPipline("./src/shaders/attentionShader/ComputeK.wgsl")
        this.sumShader = this.gpu.getShaderPipline("./src/shaders/attentionShader/sumGradient.wgsl")
    }
    async forward(vec) {
        if (this.input) await this.input.destroy()
        this.input = vec;
        const tokenCount = vec.size / (this.embeddingSize * 4)
        if (!this.asParams) this.asParams = await this.gpu.createUniFormBuffer(4 * 3);
        if (!this.softmaxParams) this.softmaxParams = await this.gpu.createUniFormBuffer(4 * 3);
        const encoder = await this.gpu.CreateEncoder()
        const scroes = await this.gpu.createBuffer(tokenCount * tokenCount * 4)
        await this.gpu.WriteBuffer(this.asParams, new Uint32Array([tokenCount, this.embeddingSize, 0]))
        await this.gpu.WriteBuffer(this.softmaxParams, new Uint32Array([tokenCount, tokenCount, 0]))
        if (this.query) this.query.destroy()
        if (this.key) this.key.destroy()
        if (this.value) this.value.destroy()
        this.query = await this.q.forward(vec)
        this.key = await this.k.forward(vec)
        this.value = await this.v.forward(vec)
        const bindAttention = await this.gpu.createBindGroup(this.attentionscoresShader, 0, [this.query, this.key, scroes, this.asParams])
        await this.gpu.RunPipline(encoder, this.attentionscoresShader, bindAttention, Math.ceil(tokenCount / 16), Math.ceil(tokenCount / 16))
        if (this.weights) this.weights.destroy()
        this.weights = await this.gpu.createBuffer(tokenCount * tokenCount * 4)
        const bindSF = await this.gpu.createBindGroup(this.softmaxForward, 0, [scroes, this.weights, this.softmaxParams]);
        await this.gpu.RunPipline(encoder, this.softmaxForward, bindSF, tokenCount)
        const finalOutput = await this.gpu.createBuffer(tokenCount * this.embeddingSize * 4)
        const outputBind = await this.gpu.createBindGroup(this.outputForward, 0, [this.weights, this.value, finalOutput, this.asParams]);
        await this.gpu.RunPipline(encoder, this.outputForward, outputBind, Math.ceil((tokenCount * this.embeddingSize) / 256))
        await this.gpu.submitQueue(encoder)
        scroes.destroy()
        return finalOutput
    }
    async backward(dO, lr) {
        const tokenCount = dO.size / (this.embeddingSize * 4);
        const encoder = await this.gpu.CreateEncoder();
        const dV = await this.gpu.createBuffer(dO.size);
        const DvBind = await this.gpu.createBindGroup(this.dVShader, 0, [this.weights, dO, dV, this.asParams]);
        await this.gpu.RunPipline(encoder, this.dVShader, DvBind, Math.ceil(tokenCount / 16), Math.ceil(this.embeddingSize / 16));
        const dA = await this.gpu.createBuffer(tokenCount * tokenCount * 4)
        const DaBind = await this.gpu.createBindGroup(this.dAshader, 0, [dO, this.value, dA, this.asParams]);
        await this.gpu.RunPipline(encoder, this.dAshader, DaBind, Math.ceil(tokenCount / 16), Math.ceil(tokenCount / 16))
        const dS = await this.gpu.createBuffer(tokenCount * tokenCount * 4);
        const dSBind = await this.gpu.createBindGroup(this.softmaxBackward, 0, [this.weights, dA, dS, this.asParams])
        await this.gpu.RunPipline(encoder, this.softmaxBackward, dSBind, Math.ceil((tokenCount * tokenCount) / 256))
        const dQ = await this.gpu.createBuffer(tokenCount * this.embeddingSize * 4);
        const dQbind = await this.gpu.createBindGroup(this.dQshader, 0, [dS, this.key, dQ, this.asParams]);
        await this.gpu.RunPipline(encoder, this.dQshader, dQbind, Math.ceil(tokenCount / 16), Math.ceil(this.embeddingSize / 16))
        const dK = await this.gpu.createBuffer(this.embeddingSize * tokenCount * 4)
        const dKbind = await this.gpu.createBindGroup(this.dKShader, 0, [dS, this.query, dK, this.asParams]);
        await this.gpu.RunPipline(encoder, this.dKShader, dKbind, Math.ceil(tokenCount / 16), Math.ceil(this.embeddingSize / 16))
        await this.gpu.submitQueue(encoder);
        const dq = await this.q.backward(dQ, lr)
        const dk = await this.k.backward(dK, lr)
        const dv = await this.v.backward(dV, lr)
        const output = await this.gpu.createBuffer(this.embeddingSize * tokenCount * 4);
        const encode = this.gpu.CreateEncoder()
        const outputBind = await this.gpu.createBindGroup(this.sumShader, 0 , [dq, dk, dv, output, this.softmaxParams]);
        await this.gpu.RunPipline(encode, this.sumShader, outputBind, Math.ceil((tokenCount * this.embeddingSize / 256)));
        await this.gpu.submitQueue(encode)
        dS.destroy()
        dA.destroy()
        dq.destroy()
        dk.destroy()
        dv.destroy()
        dK.destroy()
        dQ.destroy()
        dV.destroy()
        dO.destroy()
        return output
        
    }
    async ClearInputCache(){
       await this.q.ClearInputCache()
       await this.k.ClearInputCache()
       await this.v.ClearInputCache()
    }
    async Save(){
       await this.q.Save()
       await this.k.Save()
       await this.v.Save()
    }
}