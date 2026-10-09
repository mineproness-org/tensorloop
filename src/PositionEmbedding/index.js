import { OperationManager } from "../OperationManager.js"
import { getVector, SaveVector } from "../getVec.js"
function GenerateVectors(embeddingSize, contextSize) {
    const vec = new Float32Array(embeddingSize * contextSize).map(() => (Math.random() * 2 - 1) * 0.02)
    return vec
}

export class PositionEmbedding {
    vectors;
    vecBuffer;
    gpu;
    embeddingSize;
    contextSize;
    params;
    forwardShader;
    backwardShader;
    configs;
    constructor(device, embeddingSize, contextSize, configs) {
        this.embeddingSize = embeddingSize;
        const total = embeddingSize * contextSize;
        this.configs = configs;
        this.contextSize = contextSize;
        if (configs && configs.save) {
            const vec = getVector(configs.save.filename, total);
            if (vec) {
                this.vectors = vec;

            } else {
                this.vectors = GenerateVectors(embeddingSize, contextSize)
                SaveVector(this.configs.save.filename, this.vectors)
            }

        } else {
            this.vectors = GenerateVectors(embeddingSize, contextSize)

        }
        this.gpu = new OperationManager(device)
        this.forwardShader = this.gpu.getShaderPipline("./src/shaders/PF.wgsl")
        this.backwardShader = this.gpu.getShaderPipline("./src/shaders/PB.wgsl")
    }
    async forward(input) {
        const encoder = this.gpu.CreateEncoder()
        if (!this.params) this.params = await this.gpu.createUniFormBuffer(4 * 4)
        if (!this.vecBuffer) {
            this.vecBuffer = await this.gpu.createBuffer(Float32Array.BYTES_PER_ELEMENT * this.contextSize * this.embeddingSize);
            await this.gpu.WriteBuffer(this.vecBuffer, this.vectors)
        }
        const output = await this.gpu.createBuffer(input.size);
        const size = Math.ceil((input.size / 4) / 64);
        const bindGroup = await this.gpu.createBindGroup(this.forwardShader, 0, [this.vecBuffer, input, output])
        if (this.prevVec) this.prevVec.destroy();
        this.prevVec = input;
        await this.gpu.RunPipline(encoder, this.forwardShader, bindGroup, size);
        await this.gpu.submitQueue(encoder);
        return output
    }
    async backward(outputGradient, lr){
        const encoder = await this.gpu.CreateEncoder()
        await this.gpu.WriteBuffer(this.params, new Float32Array([lr,0]))
        const bindGroup = await this.gpu.createBindGroup(this.backwardShader, 0, [this.vecBuffer, outputGradient, this.params])
        if(this.prevGradient) await this.prevGradient.destroy()
        this.prevGradient = outputGradient;
        await this.gpu.RunPipline(encoder, this.backwardShader, bindGroup, Math.ceil((outputGradient.size / 4) / 64))
        await this.gpu.submitQueue(encoder);
    }
    async Save() {
        const vec = await this.gpu.readBuffer(Float32Array.BYTES_PER_ELEMENT * this.contextSize * this.embeddingSize, this.vecBuffer)
        SaveVector(this.configs.save.filename, vec)
    }
}