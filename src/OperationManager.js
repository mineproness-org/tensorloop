import fs from 'fs'
import path, { join, dirname, format } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url))

export class OperationManager {
    device;
    matrixMatSHhaders;
    params;
    addShader;
    constructor(device) {
        this.device = device;
        this.addShader = this.getShaderPipline("./src/shaders/add.wgsl")
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

    async MakeACopyBuffer(encoder2, sourceBuffer) {
         const encoder = encoder2 ? encoder2 : await this.CreateEncoder()
        const buffer = this.device.createBuffer({
            size: sourceBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE
        })
        encoder.copyBufferToBuffer(sourceBuffer, 0, buffer, 0, sourceBuffer.size);
        if(!encoder2) await this.submitQueue(encoder)
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
    async add(a, b, size) {
        if (!this.params) {
            this.params = await this.createUniFormBuffer(4 * 4)

        }
        const encoder = await this.CreateEncoder()
        const output = await this.createBuffer(a.size);
        const bind = await this.createBindGroup(this.addShader, 0, [a, b, output, this.params]);
        await this.WriteBuffer(this.params, new Uint32Array([size, 0]))
        await this.RunPipline(encoder, this.addShader, bind, Math.ceil(size / 64));
        await this.submitQueue(encoder)
        // a.destroy()
        // b.destroy()
        return output
    }

}   