import { OperationManager } from '../OperationManager.js'
import fs from 'fs'

export class RMSNorm {
  embeddingSize;
  reductionShader;
  forwardShader;
  backwardShader;
  op;
  constructor(device, embeddingSize) {
    this.op = new OperationManager(device);
    this.embeddingSize = embeddingSize;
    this.reductionShader = this.op.getShaderPipline("./src/shaders/Norm/sumSquare.wgsl")
    this.forwardShader = this.op.getShaderPipline("./src/shaders/Norm/rmsForward.wgsl")
    this.backwardShader = this.op.getShaderPipline("./src/shaders/Norm/rmsBackward.wgsl")
  }
  async forward(input) {
    const outputReduce = await this.op.createBuffer(4 * 1);
    const output = await this.op.createBuffer(input.size);
    const bindGroup = await this.op.createBindGroup(this.reductionShader, 0, [input, outputReduce])
    const encoder = await this.op.CreateEncoder()
    await this.op.RunPipline(encoder, this.reductionShader, bindGroup, Math.ceil((input.size / 4) / 64))
    const bindGroup1 = await this.op.createBindGroup(this.forwardShader, 0, [outputReduce, input, output])
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
  async backward(outputGradient) {
    const output = await this.op.createBuffer(outputGradient.size);
    const bindGroup = await this.op.createBindGroup(this.backwardShader, 0, [this.input, outputGradient, this.rms, output]);
    const encoder = await this.op.CreateEncoder()
    await this.op.RunPipline(encoder, this.backwardShader, bindGroup, Math.ceil((this.input.size / 4) / 64))
    await this.op.submitQueue(encoder)
    if (this.prevOut) this.prevOut.destroy()
    this.prevOut = outputGradient
    return output
  }

}