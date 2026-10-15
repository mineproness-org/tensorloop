import { OperationManager } from '../OperationManager.js'
import fs from 'fs'
import { SaveVector, getVector } from '../getVec.js'

export class RMSNorm {
  embeddingSize;
  reductionShader;
  forwardShader;
  backwardShader;
  op;
  gamma;
  gammabuffer;
  configs;
  constructor(device, embeddingSize, configs) {
    this.configs = configs
    this.op = new OperationManager(device);
    this.embeddingSize = embeddingSize;
    this.reductionShader = this.op.getShaderPipline("./src/shaders/Norm/sumSquare.wgsl", {
      embeddingSize: embeddingSize,
    })
    this.forwardShader = this.op.getShaderPipline("./src/shaders/Norm/rmsForward.wgsl", {
      embeddingSize: embeddingSize
    })
    this.backwardShader = this.op.getShaderPipline("./src/shaders/Norm/rmsBackward.wgsl", {
      embeddingSize: embeddingSize
    })
    this.updateGamma = this.op.getShaderPipline("./src/shaders/Norm/updateGamma.wgsl", {
      embeddingSize: embeddingSize
    })
    if (configs) {
      const ga = getVector(configs.save.filename, embeddingSize);
      if (!ga) {
        this.gamma = new Float32Array(embeddingSize).fill(1);
        SaveVector(configs.save.filename, this.gamma)
      } else {
        this.gamma = ga
      }
    } else {
      this.gamma = new Float32Array(embeddingSize).fill(1)
    }
  }
  async forward(input) {
    if (!this.gammabuffer) {
      this.gammabuffer = await this.op.createBuffer(4 * this.embeddingSize);
      await this.op.WriteBuffer(this.gammabuffer, this.gamma);
    }
    let context = input.size / (this.embeddingSize * 4)
    const outputReduce = await this.op.createBuffer(4 * context);
    const output = await this.op.createBuffer(input.size);
    const bindGroup = await this.op.createBindGroup(this.reductionShader, 0, [input, outputReduce])
    const encoder = await this.op.CreateEncoder()
    await this.op.RunPipline(encoder, this.reductionShader, bindGroup, context)
    const bindGroup1 = await this.op.createBindGroup(this.forwardShader, 0, [outputReduce, input, output, this.gammabuffer])
    await this.op.RunPipline(encoder, this.forwardShader, bindGroup1, Math.ceil((input.size / 4) / 64));
    await this.op.submitQueue(encoder);
    if (this.rms) this.rms.destroy()
    this.rms = outputReduce
    if (this.input) this.input.destroy()
    this.input = input;
    if (this.outputPre) this.outputPre.destroy()
    this.outputPre = output
    return output
  }
  async backward(outputGradient, lr) {
    const output = await this.op.createBuffer(outputGradient.size);
    let params = await this.op.createBuffer(4 * 2)
    await this.op.WriteBuffer(params, new Float32Array([lr, 0]))
    const bindGroup = await this.op.createBindGroup(this.backwardShader, 0, [this.input, outputGradient, this.rms, this.gammabuffer, output]);
    const bindGroup2 = await this.op.createBindGroup(this.updateGamma, 0, [this.input, outputGradient, this.rms, this.gammabuffer, params]);
    const encoder = await this.op.CreateEncoder()
    await this.op.RunPipline(encoder, this.backwardShader, bindGroup, Math.ceil((this.input.size) / (this.embeddingSize * 4)))
    await this.op.RunPipline(encoder, this.updateGamma, bindGroup2, Math.ceil(this.embeddingSize / 64))
    await this.op.submitQueue(encoder)
    if (this.prevOut) this.prevOut.destroy()
    this.prevOut = outputGradient
    params.destroy()
    return output
  }
  async Save(){
    const read = await this.op.readBuffer(4 * this.embeddingSize, this.gammabuffer);
    SaveVector(this.configs.save.filename, read)
  }

}