import { log } from 'console';
import { OperationManager } from '../OperationManager.js'

export class Softmax {
    device;
    vocabSize;
    forwardPipline;
    embeddingSize;
    gpu;
    paramsForward;
    oldParam;
    bp
    constructor(device, embeddingSize, vocabSize) {
        this.vocabSize = vocabSize;
        this.device = device;
        this.embeddingSize = embeddingSize;
        this.gpu = new OperationManager(device);
        this.forwardPipline = this.gpu.getShaderPipline("./src/shaders/SF.wgsl")
        this.backwardPipline = this.gpu.getShaderPipline("./src/shaders/SB.wgsl")
        this.lossPipline = this.gpu.getShaderPipline("./src/shaders/CalculateLoss.wgsl")
        this.accPipline = this.gpu.getShaderPipline("./src/shaders/CalculateAcc.wgsl")
        this.battention = this.gpu.getShaderPipline("./src/shaders/SoftmaxBackward.wgsl")
        this.reduceFn = this.gpu.getShaderPipline("./src/shaders/ReductionFunction.wgsl")
    }
    async forward(logits, cpuReadBack, encoderCPU) {
        const encoder = encoderCPU ? encoderCPU : this.device.createCommandEncoder()
        const tokenCount = logits.size / (this.vocabSize * 4)
        if (!this.paramsForward) {
            this.paramsForward = await this.gpu.createUniFormBuffer(4 * 3)
        }
        if (!this.oldParam) {
            this.oldParam = new Uint32Array([this.vocabSize, 0, 0])
        }

        if (this.oldParam[1] !== tokenCount) { await this.gpu.WriteBuffer(this.paramsForward, new Uint32Array([this.vocabSize, tokenCount, 0])); this.oldParam = new Uint32Array([this.vocabSize, tokenCount, 0]); }
        const output = await this.gpu.createBuffer(logits.size);
        const bind = await this.device.createBindGroup({
            layout: await this.forwardPipline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: logits
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: output
                    }
                }, {
                    binding: 2,
                    resource: {
                        buffer: this.paramsForward
                    }
                }
            ]
        })
        await this.gpu.RunPipline(encoder, this.forwardPipline, bind, Math.ceil((tokenCount * this.vocabSize) / 256))
        if (!encoderCPU) {
            await this.device.queue.submit([encoder.finish()])
            logits.destroy()
        }
        if (cpuReadBack) {
            const outputCPU = await this.gpu.readBuffer(output.size, output);
            return outputCPU
        }
        return output
    }
    async backward(probs, target) {
        const encoder = this.device.createCommandEncoder()
        const tokenCount = probs.size / (this.vocabSize * 4)
        if (!this.paramsForward) {
            this.paramsForward = await this.gpu.createUniFormBuffer(4 * 3)
        }
        if (!this.oldParam) {
            this.oldParam = new Uint32Array([this.vocabSize, 0, 0])
        }
        if (this.oldParam[1] !== tokenCount) {
            await this.gpu.WriteBuffer(this.paramsForward, new Uint32Array([this.vocabSize, tokenCount, 0]));
            this.oldParam = new Uint32Array([this.vocabSize, tokenCount, 0]);
        }
        const targetBuffer = await this.gpu.createBuffer(target.length * Float32Array.BYTES_PER_ELEMENT)
        await this.gpu.WriteBuffer(targetBuffer, new Float32Array(target))
        const output = await this.gpu.createBuffer(probs.size)
        const bind = await this.device.createBindGroup({
            layout: await this.backwardPipline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: probs
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: output
                    }
                }, {
                    binding: 2,
                    resource: {
                        buffer: targetBuffer
                    }
                }, {
                    binding: 3,
                    resource: {
                        buffer: this.paramsForward
                    }
                }
            ]
        })
        await this.gpu.RunPipline(encoder, this.backwardPipline, bind, Math.ceil((tokenCount * this.vocabSize) / 256))
        this.device.queue.submit([encoder.finish()])
        probs.destroy()
        targetBuffer.destroy()
        return output
    }
    async crossEntropy(probs, target) {
        const encoder = this.device.createCommandEncoder()

        const tokenCount = probs.size / (this.vocabSize * 4)
        const targetBuffer = await this.gpu.createBuffer(target.length * Float32Array.BYTES_PER_ELEMENT)
        const loss = await this.gpu.createBuffer(4 * tokenCount)
        const acc = await this.gpu.createBuffer(4 * 2)
        await this.gpu.WriteBuffer(targetBuffer, new Float32Array(target))
        const bind = await this.device.createBindGroup({
            layout: await this.lossPipline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: probs
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: targetBuffer
                    }
                }, {
                    binding: 2,
                    resource: {
                        buffer: loss
                    }
                }, {
                    binding: 3,
                    resource: {
                        buffer: this.paramsForward
                    }

                },
            ]
        })
        await this.gpu.RunPipline(encoder, this.lossPipline, bind, Math.ceil((tokenCount / 4)))
        const outputLoss = await this.gpu.createBuffer(4)
        const reduceBind = await this.gpu.createBindGroup(this.reduceFn, 0, [loss, outputLoss, this.paramsForward])
        await this.gpu.RunPipline(encoder, this.reduceFn, reduceBind, 1);
        const bind2 = await this.device.createBindGroup({
            layout: await this.accPipline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.paramsForward
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: probs
                    }
                }, {
                    binding: 3,
                    resource: {
                        buffer: targetBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: acc
                    }
                }
            ]
        })
        await this.gpu.RunPipline(encoder, this.accPipline, bind2, 1)
        this.device.queue.submit([encoder.finish()])
        targetBuffer.destroy()
        loss.destroy()
        const lossCpu = await this.gpu.readBuffer(outputLoss.size, outputLoss)
        const accCPU = await this.gpu.readBuffer(acc.size, acc)
        acc.destroy()
        return { loss: lossCpu[0], acc: accCPU[0] }
    }

}

