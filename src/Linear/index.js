
import { create, globals } from "webgpu";
import fs from 'fs'
import { buffer } from "stream/consumers";
import { SaveVector, getVector } from '../getVec.js'
import { OperationManager } from "../OperationManager.js";
Object.assign(globalThis, globals)
const navigator = { gpu: create([]) }


function GenerateVectors(embeddingSize, vocabSize) {
    const vec = new Float32Array(embeddingSize * vocabSize).map(() => (Math.random() * 2 - 1) * 0.02)
    return vec
}

function GenerateBias(vocabSize) {
    const vec = new Float32Array(vocabSize)
    return vec
}

export class Linear {
    device;
    embeddingSize;
    vocabSize;
    configs;
    bias;
    weights;
    forwardPipline;
    configs;
    input;
    wBuffer;
    bBuffer;
    gpu;
    paramForward;
    oldParam;
    constructor(device, embeddingSize, vocabSize, configs) {
        this.configs = configs
        this.embeddingSize = embeddingSize;
        this.vocabSize = vocabSize;
        this.configs = configs;
        this.device = device
        let total = embeddingSize * vocabSize;
        if (configs && configs.save) {
            const w = getVector(configs.save.filename[0], total);
            const b = getVector(configs.save.filename[1], total, true)
            if (!w && !b) {
                this.bias = GenerateBias(vocabSize)
                this.weights = GenerateVectors(embeddingSize, vocabSize)
                SaveVector(configs.save.filename[0], this.weights)
                SaveVector(configs.save.filename[1], this.bias)
            } else {
                this.bias = b;
                this.weights = w;
            }
        } else {
            this.bias = GenerateBias(vocabSize)
            this.weights = GenerateVectors(embeddingSize, vocabSize)
        }
        this.gpu = new OperationManager(device);
        this.forwardPipline = this.gpu.getShaderPipline("./src/shaders/LF.wgsl")
        this.backwardPipline = this.gpu.getShaderPipline("./src/shaders/LB.wgsl")
        this.dInputBackward = this.gpu.getShaderPipline("./src/shaders/ComputeDInput.wgsl")
    }
    async forward(vec) {
        const encoder = this.device.createCommandEncoder()
        const tokenCount = vec.size / (this.embeddingSize * 4)
        if (!this.wBuffer) {

            this.wBuffer = await this.gpu.createBuffer(this.embeddingSize * this.vocabSize * 4)
            await this.gpu.WriteBuffer(this.wBuffer, this.weights)
            this.bBuffer = await this.gpu.createBuffer(this.vocabSize * 4)
            await this.gpu.WriteBuffer(this.bBuffer, this.bias)
            this.paramForward = await this.gpu.createUniFormBuffer(4 * 4)
            this.paramBackward = await this.gpu.createUniFormBuffer(4 * 5)
        }
        if (!this.oldParam) {
            this.oldParam = new Uint32Array([this.embeddingSize, this.vocabSize, 0, 0])
        }
        if (this.oldParam[2] !== tokenCount) {
            await this.gpu.WriteBuffer(this.paramForward, new Uint32Array([this.embeddingSize, this.vocabSize, tokenCount, 0]))
            this.oldParam = new Uint32Array([this.embeddingSize, this.vocabSize, tokenCount, 0]);
        }
        const outputBuffer = await this.gpu.createBuffer(4 * this.vocabSize * tokenCount)
        const bind = await this.device.createBindGroup({
            layout: await this.forwardPipline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: this.wBuffer
                },
                {
                    binding: 1,
                    resource: outputBuffer
                },
                {
                    binding: 2,
                    resource: vec
                },
                {
                    binding: 3,
                    resource: this.bBuffer
                },
                {
                    binding: 4,
                    resource: this.paramForward
                }
            ]
        })
        await this.gpu.RunPipline(encoder, this.forwardPipline, bind, Math.ceil((this.vocabSize * tokenCount) / 256))
        if (this.input) this.input.destroy()
        this.input = await this.gpu.MakeACopyBuffer(encoder, vec);
        this.device.queue.submit([encoder.finish()])
        return outputBuffer
    }
    async backward(dOutput, learningRate) {
        const encoder = this.device.createCommandEncoder()
        const tokenCount = dOutput.size / (this.vocabSize * 4)
        if (!this.oldParamb) {
            this.oldParamb = new Float32Array([this.vocabSize, 0, 0, this.embeddingSize, 0])
        }
        if (this.oldParamb[1] !== tokenCount || this.oldParamb[2] !== learningRate) {
            await this.gpu.WriteBuffer(this.paramBackward, new Float32Array([this.vocabSize, tokenCount, learningRate, this.embeddingSize, 0]));
            this.oldParamb = new Float32Array([this.vocabSize, tokenCount, learningRate, this.embeddingSize, 0]);
        }
        const outputBuffer = await this.gpu.createBuffer(this.embeddingSize * tokenCount * 4);
        const bind = await this.device.createBindGroup({
            layout: await this.backwardPipline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: this.bBuffer
                },
                {
                    binding: 1,
                    resource: this.wBuffer
                },
                {
                    binding: 2,
                    resource: dOutput
                },
                {
                    binding: 3,
                    resource: this.input
                },
                {
                    binding: 4,
                    resource: this.paramBackward
                }
            ]
        })
        const bind2 = await this.device.createBindGroup({
            layout: await this.dInputBackward.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: this.wBuffer
                },
                {
                    binding: 1,
                    resource: dOutput
                },
                {
                    binding: 3,
                    resource: this.paramBackward
                },
                {
                    binding: 4,
                    resource: outputBuffer
                }
            ]
        })
        // console.log((await this.gpu.readBuffer(this.input.size, this.input))[0])
        await this.gpu.RunPipline(encoder, this.dInputBackward, bind2, Math.ceil((this.embeddingSize * tokenCount) / 256));
        await this.gpu.RunPipline(encoder, this.backwardPipline, bind, Math.ceil((this.vocabSize * tokenCount) / 256));
        await this.device.queue.submit([encoder.finish()])
        await dOutput.destroy()
        this.input.destroy()
        return outputBuffer

    }
    async Save() {
        const b = await this.gpu.readBuffer(this.bBuffer.size, this.bBuffer);
        const w = await this.gpu.readBuffer(this.wBuffer.size, this.wBuffer);
        SaveVector(this.configs.save.filename[0], w)
        SaveVector(this.configs.save.filename[1], b)
    }
    async ClearInputCache() {
        if (this.input) await this.input.destroy();
    }

}