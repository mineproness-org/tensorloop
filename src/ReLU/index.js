import { OperationManager } from "../OperationManager.js";

export class Relu {
    embeddingSize;
    paramBuffer;
    oldParam;
    forwardPipline;
    gpu;
    backwardPipline;
    input;
    constructor(device, embeddingSize) {
        this.embeddingSize = embeddingSize
        this.gpu = new OperationManager(device);
        this.device = device;
        this.forwardPipline = this.gpu.getShaderPipline("./src/shaders/RF.wgsl")
        this.backwardPipline = this.gpu.getShaderPipline("./src/shaders/RB.wgsl")
    }
    async forward(input) {
        const encoder = this.device.createCommandEncoder()
        const tokenCount = input.size / (this.embeddingSize * 4)
        if (!this.oldParam) {
            this.oldParam = new Uint32Array([0, 0, 0]);
            this.paramBuffer = await this.gpu.createUniFormBuffer(4 * 3);
        }
        const output = await this.gpu.createBuffer(input.size)
        if (this.oldParam[1] !== tokenCount) {
            this.oldParam = new Uint32Array([this.embeddingSize, tokenCount, 0])
            await this.gpu.WriteBuffer(this.paramBuffer, this.oldParam)
        }
        const bind = await this.device.createBindGroup({
            layout: await this.forwardPipline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: input
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.paramBuffer
                    }
                }, {
                    binding: 2,
                    resource: {
                        buffer: output
                    }
                }
            ]
        })
        await this.gpu.RunPipline(encoder, this.forwardPipline, bind, Math.ceil((tokenCount * this.embeddingSize) / 256))
        this.device.queue.submit([encoder.finish()])
        if(this.input) this.input.destroy()
        this.input = input
        return output
    }
    async backward(inputGradient) {
        const encoder = this.device.createCommandEncoder()
        const tokenCount = inputGradient.size / (this.embeddingSize * 4)
        if (!this.oldParam) {
            this.oldParam = new Uint32Array([0, 0, 0]);
            this.paramBuffer = await this.gpu.createUniFormBuffer(4 * 3);
        }
        const output = await this.gpu.createBuffer(inputGradient.size)
        if (this.oldParam[1] !== tokenCount) {
            this.oldParam = new Uint32Array([this.embeddingSize, tokenCount, 0])
            await this.gpu.WriteBuffer(this.paramBuffer, this.oldParam)
        }
        const bind = await this.device.createBindGroup({
            layout: await this.backwardPipline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.input
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.paramBuffer
                    }
                }, {
                    binding: 2,
                    resource: {
                        buffer: output
                    }
                }, {
                    binding: 3,
                    resource: {
                        buffer: inputGradient
                    }
                }
            ]
        })
        await this.gpu.RunPipline(encoder, this.backwardPipline, bind, Math.ceil((tokenCount * this.embeddingSize) / 256))
        await this.device.queue.submit([encoder.finish()])
        this.input.destroy()
        inputGradient.destroy()
        return output
    }
}