
import { SelfAttention } from '../SelfAttention/index.js';
import { FFN } from '../FFN/index.js';
import { existsSync, mkdirSync } from 'fs';
import { RMSNorm } from '../../index.js';
import { OperationManager } from '../OperationManager.js';
import { join } from 'path';

export class TransformerBlock {
    attention;
    ffn;
    gpu;
    size;
    rmsnorm;
    rmsnorm2;

    constructor(device, embeddingSize, configs) {
        this.size = embeddingSize;

        this.rmsnorm = new RMSNorm(device, embeddingSize, {
            save: {
                filename: join(configs.dirname, "LAYERNORM-1.bin")
            }
        });
        this.rmsnorm2 = new RMSNorm(device, embeddingSize, {
            save: {
                filename: join(configs.dirname, "LAYERNORM-2.bin")
            }
        });

        if (!existsSync(configs.dirname)) {
            mkdirSync(configs.dirname, { recursive: true });
        }

        this.attention = new SelfAttention(
            device, embeddingSize, configs
        );

        this.ffn = new FFN(device, embeddingSize, configs);
        this.gpu = new OperationManager(device);
    }

    async forward(input) {
        const count = input.size / 4;
        const norm1 = await this.rmsnorm.forward(input);
        const att = await this.attention.forward(norm1);
        const h = await this.gpu.add(input, att, count);
        const norm2 = await this.rmsnorm2.forward(h);
        const f = await this.ffn.forward(norm2);
        const y = await this.gpu.add(h, f, count);
        norm1.destroy();
        att.destroy();
        f.destroy();
        return y;
    }

    async backward(outGradient, lr) {
        const count = outGradient.size / 4;
        const dNorm2 = await this.ffn.backward(outGradient, lr);
        const dHFromFFN = await this.rmsnorm2.backward(dNorm2 , lr);
        const dH = await this.gpu.add(outGradient, dHFromFFN, count);
        const dNorm1 = await this.attention.backward(dH, lr);
        const dInputFromAttention = await this.rmsnorm.backward(dNorm1, lr);
        const dInput = await this.gpu.add(dH, dInputFromAttention, count);
        dNorm2.destroy();
        dHFromFFN.destroy();
        dH.destroy();
        dNorm1.destroy();
        dInputFromAttention.destroy();
        dInputFromAttention.destroy();
        return dInput;
    }

    async ClearInputCache() {
        await this.ffn.ClearInputCache();
        await this.attention.ClearInputCache();
    }

    async Save() {
        await this.ffn.Save();
        await this.attention.Save();
        await this.rmsnorm.Save()
        await this.rmsnorm2.Save()
    }
}