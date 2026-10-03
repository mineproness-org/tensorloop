import { config } from "process";

import { SaveVector, getVector } from '../getVec.js'
import { OperationManager } from "../OperationManager.js";

function GenerateVectors(embeddingSize, vocabSize) {
    const vec = new Float32Array(embeddingSize * vocabSize).map(() => (Math.random() * 2 - 1) * 0.02)
    return vec
}

export class Embedding {
    embeddingSize;
    vocabSize;
    gpu;
    firstTrue;
    vectors;
    configs;
    vectorsBuffer;
    forwardPipline;
    constructor(device, embeddingSize, vocabSize, configs) {
        this.configs = configs
        this.embeddingSize = embeddingSize;
        this.vocabSize = vocabSize;
        this.device = device
        let total = embeddingSize * vocabSize;
        if (configs && configs.save) {
            const vec = getVector(configs.save.filename, total);
            if (vec) {
                this.vectors = vec;

            } else {
                this.vectors = GenerateVectors(embeddingSize, vocabSize)
            }

        } else {
            this.vectors = GenerateVectors(embeddingSize, vocabSize)

        }
        this.gpu = new OperationManager(device)
        this.forwardPipline = this.gpu.getShaderPipline("./src/shaders/EF.wgsl", {
            embeddingSize: embeddingSize
        })
        this.backwardPipline = this.gpu.getShaderPipline("./src/shaders/EB.wgsl", {
            embeddingSize: embeddingSize
        })
        SaveVector(this.configs.save.filename, this.vectors)
    }
    async forward(input) {
        const encoder = this.device.createCommandEncoder()
        if (!this.vectorsBuffer) {
            this.vectorsBuffer = await this.gpu.createBuffer(Float32Array.BYTES_PER_ELEMENT * this.vocabSize * this.embeddingSize);
            await this.gpu.WriteBuffer(this.vectorsBuffer, this.vectors)
        }
        const tokenBuffer = await this.gpu.createBuffer(Uint32Array.BYTES_PER_ELEMENT * input.length);
        await this.gpu.WriteBuffer(tokenBuffer, new Uint32Array(input))
        const outputBuffer = await this.gpu.createBuffer(Float32Array.BYTES_PER_ELEMENT * this.embeddingSize * input.length);
        const bind = await this.device.createBindGroup({
            layout: await this.forwardPipline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.vectorsBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: outputBuffer
                    }
                }, {
                    binding: 2,
                    resource: {
                        buffer: tokenBuffer
                    }
                }
            ]
        })
        await this.gpu.RunPipline(encoder,this.forwardPipline, bind, Math.ceil((this.embeddingSize * input.length) / 256))
        this.device.queue.submit([encoder.finish()])
        tokenBuffer.destroy()
        if (this.configs.cpuReadBack) {
            const output = await this.gpu.readBuffer(Float32Array.BYTES_PER_ELEMENT * this.embeddingSize * input.length, outputBuffer);
            outputBuffer.destroy()
            return output
        }
        return outputBuffer

    }
    async Save() {
        const vec = await this.gpu.readBuffer(Float32Array.BYTES_PER_ELEMENT * this.vocabSize * this.embeddingSize, this.vectorsBuffer)
        SaveVector(this.configs.save.filename, vec)
    }
    async backward(token, inputGradient, lr) {
        const encoder = this.device.createCommandEncoder()
        if (!this.oldParam) {
            this.oldParam = new Float32Array([0, 0]);
            this.param = await this.gpu.createUniFormBuffer(this.oldParam.byteLength);
            this.tokenBuffer = await this.gpu.createBuffer(4 * token.length)
        }
        if (this.oldParam[1] !== lr) {
            this.oldParam = new Float32Array([lr, 0]);
            this.gpu.WriteBuffer(this.param, this.oldParam)
        }
        const bind = await this.device.createBindGroup({
            layout: await this.backwardPipline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.vectorsBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: inputGradient
                    }
                }, {
                    binding: 2,
                    resource: {
                        buffer: this.tokenBuffer
                    }

                }, {
                    binding: 3,
                    resource: {
                        buffer: this.param
                    }

                }
            ]
        })
        await this.gpu.WriteBuffer(this.tokenBuffer, new Uint32Array(token));
        await this.gpu.RunPipline(encoder, this.backwardPipline, bind, Math.ceil((this.embeddingSize * token.length) / 256))
        this.device.queue.submit([encoder.finish()])
        inputGradient.destroy()
    }
}