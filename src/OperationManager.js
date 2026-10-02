import fs from 'fs'
import path, { join, dirname, format } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url))

export class OperationManager {
    device;
    matrixMatSHhaders;
    params;
    matrixParams;
    constructor(device) {
        this.device = device;
    }
    async createBuffer(size) {
        return this.device.createBuffer({
            size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE
        })
    }
    async createUniFormBuffer(size) {
        return this.device.createBuffer({
            size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.UNIFORM
        })
    }
    async readBuffer(size, buffer) {
        const readBuffer = this.device.createBuffer({
            size: size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        })
        const encoder = this.device.createCommandEncoder()
        encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
        this.device.queue.submit([await encoder.finish()])
        await readBuffer.mapAsync(GPUMapMode.READ);
        const mapped = readBuffer.getMappedRange();
        const result = new Float32Array(
            mapped.slice(0)
        );
        readBuffer.unmap()
        readBuffer.destroy()
        return result
    }
    getShaderPipline(filename, value = { "embedding": 123 }, entryPoint = "main") {
        let readShaderCode = fs.readFileSync(path.join(__dirname, `../${filename}`), "utf-8");
        if (value) {
            for (const a in value) {
                readShaderCode = readShaderCode.replaceAll(`%${a}%`, value[a])
            }
        };
        const compiledKernel = this.device.createShaderModule({
            code: readShaderCode
        });
        const compiledPipLine = this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: compiledKernel,
                entryPoint: entryPoint
            }
        });
        return compiledPipLine;
    }
    async WriteBuffer(buffer, data) {
        this.device.queue.writeBuffer(buffer, 0, data)
    }
    async RunPipline(encoder, pipline, binding, dispatchWorkgroups, y = 1, z = 1) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipline)
        pass.setBindGroup(0, binding)
        pass.dispatchWorkgroups(dispatchWorkgroups, y, z)
        pass.end()
    }

    async MakeACopyBuffer(encoder, sourceBuffer) {
        const buffer = this.device.createBuffer({
            size: sourceBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE
        })
        encoder.copyBufferToBuffer(sourceBuffer, 0, buffer, 0, sourceBuffer.size);
        return buffer
    }
    CreateEncoder() {
        return this.device.createCommandEncoder()
    }
    createBindGroup(shader, groupNum, ent) {
        return this.device.createBindGroup({
            layout: shader.getBindGroupLayout(groupNum),
            entries: ent.map((e, x) => {
                return {
                    binding: x,
                    resource: e
                }
            })
        })
    }
    async submitQueue(encoder) {
        this.device.queue.submit([encoder.finish()])
    }

}   