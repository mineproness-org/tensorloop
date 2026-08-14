import { SelfAttention } from '../SelfAttention/index.js'
import { FFN } from '../FFN/index.js'
import { LayerNorm } from '../LayerNorm/index.js'
export class TransformerBlock {
    norm1;
    norm2;
    ffn;
    selfAttention;
    constructor(embeddingSize, configs) {
        this.norm1 = new LayerNorm(embeddingSize)
        this.norm2 = new LayerNorm(embeddingSize)
        this.ffn = new FFN(embeddingSize, "gelu", configs)
        this.selfAttention = new SelfAttention(embeddingSize, configs)
    }
    forward(input) {
        const normal1 = []
        for (let a = 0; a < input.length; a++) {
            normal1[a] = this.norm1.forward(input[a], a)
        }
        const attention = this.selfAttention.forward(normal1)
        const residual1 = []
        for (let a = 0; a < input.length; a++) {
            residual1[a] = new Float32Array(input[0].length)
            for (let i = 0; i < input[0].length; i++) {
                residual1[a][i] = input[a][i] + attention[a][i]
            }
        }
        const normal2 = []
        for (let a = 0; a < input.length; a++) {
            normal2[a] = this.norm2.forward(residual1[a], a)
        }
        const ffnOutput = this.ffn.forward(normal2)
        const output = []
        for (let g = 0; g < residual1.length; g++) {
            output[g] = new Float32Array(residual1[0].length)
            for (let a = 0; a < output[0].length; a++) {
                output[g][a] = residual1[g][a] + ffnOutput[g][a]
            }
        }
        return output
    }
    backward(dOutput, learningRate) {
        const dFFN = this.ffn.backward(
            dOutput,
            learningRate
        )
        const dResidual1 = []

        for (let a = 0; a < dFFN.length; a++) {
            dResidual1[a] = this.norm2.backward(
                dFFN[a],
                learningRate,
                a
            )
        }
        const dAttention = this.selfAttention.backward(
            dResidual1,
            learningRate
        )
        const dInputFromAttention = []
        for (let a = 0; a < dAttention.length; a++) {
            dInputFromAttention[a] = this.norm1.backward(
                dAttention[a],
                learningRate,
                a
            )
        }
        const dInput = []
        for (let a = 0; a < dOutput.length; a++) {
            dInput[a] = new Float32Array(dOutput[a].length)
            for (let i = 0; i < dOutput[a].length; i++) {
                dInput[a][i] =
                    dOutput[a][i] +
                    dInputFromAttention[a][i]
            }
        }

        return dInput
    }
    clearInputCache() {
        this.ffn.clearInputCache()
        this.selfAttention.key.ClearInputCache()
        this.selfAttention.query.ClearInputCache()
        this.selfAttention.value.ClearInputCache()
        this.norm1.ClearInputCache()
        this.norm2.ClearInputCache()
    }
    Save() {
        this.selfAttention.Save()
        this.ffn.Save()
    }
}