import { GPU } from 'gpu.js'

const gpu = new GPU({ mode: "gpu" })
if (gpu.mode == "cpu") {
    throw new Error("The tensorloop didn't have working GPU access.")
}

// ==========================
//      Init all Things
// ==========================

const maxTexture = GetMaxTexture()

// ===========================
//       UTILS FUNCITON
// ===========================
function GetMaxTexture() {
    const dummyKernel = gpu.createKernel(function () {
        return 1 + 1
    }).setOutput([1])
    dummyKernel()
    const gl = dummyKernel.context;
    return gl.getParameter(gl.MAX_TEXTURE_SIZE)
}

export function spilt2DArray(arr) {
    const result = []
    for (let a = 0; a < arr.length; a += maxTexture) {
        result.push(arr.slice(a, a + maxTexture))
    }
    return result
}
function split2DByColumns(matrix, chunkSize = maxTexture) {
    const chunks = [];
    for (let start = 0; start < matrix[0].length; start += chunkSize) {
        const end = Math.min(start + chunkSize, matrix[0].length);
        const chunk = matrix.map(row =>
            row.slice(start, end)
        );
        chunks.push(chunk);
    }
    return chunks;
}
// =======================
//    Init all Classes
// =======================


export class GPULinear {
    cacheLF = [];
    cacheLInputGra = []
    cacheLWeigths = []
    cacheLBias = []
    Weights = [];
    Bias = [];
    contextSize;
    embeddingSize;
    outputSize;
    constructor(weights, bias, contextSize) {

        this.contextSize = contextSize;
        this.embeddingSize = weights[0].length;
        this.outputSize = weights.length
        const splitW = spilt2DArray(weights)
        const splitB = spilt2DArray(bias)
        process.stdout.write(`\r[DEBUG] Saving Weights in VRAM${".".repeat(Math.floor(Math.random() * 4))}`)
        for (let i = 0; i < splitW.length; i++) {
            const biasVram = gpu.createKernel(function (x) {
                return x[this.thread.x]
            }, {
                output: [splitW[i].length],
                immutable: true,
                pipeline: true
            })
            const weightVram = gpu.createKernel(function (x) {
                return x[this.thread.y][this.thread.x]
            }, {
                output: [this.embeddingSize, splitW[i].length],
                immutable: true,
                pipeline: true
            })
            this.Weights[i] = weightVram(splitW[i]);
            this.Bias[i] = biasVram(splitB[i]);
            this.cacheLF[i] = gpu.createKernel(function (weights, bias, x) {
                let sum = bias[this.thread.x]
                for (let i = 0; i < this.constants.inputSize; i++) {
                    sum += weights[this.thread.x][i] * x[this.thread.y][i]
                }
                return sum;
            }).setOutput([splitW[i].length, contextSize]).setConstants({ inputSize: this.embeddingSize })
            this.cacheLWeigths[i] = gpu.createKernel(function (weights, learningRate, out, input) {
                let sum = 0;
                for (let a = 0; a < this.constants.contextSize; a++) {
                    sum += out[a][this.thread.y] * input[a][this.thread.x]
                }
                return weights[this.thread.y][this.thread.x] - learningRate * sum

            }, {
                constants: {
                    contextSize
                },
                output: [this.embeddingSize, splitW[i].length],
                pipeline: true,
                immutable: true
            })
            this.cacheLBias[i] = gpu.createKernel(function (bias, learningRate, out) {
                let sum = 0;
                for (let a = 0; a < this.constants.contextSize; a++) {
                    sum += out[a][this.thread.x]
                }
                return bias[this.thread.x] - learningRate * sum

            }, {
                constants: {
                    contextSize
                },
                output: [splitW[i].length],
                pipeline: true,
                immutable: true
            })
            this.cacheLInputGra[i] = gpu.createKernel(function (weights, out) {
                let sum = 0;
                for (let a = 0; a < this.constants.outputSizes; a++) {
                    sum += out[this.thread.y][a] * weights[a][this.thread.x]
                }
                return sum

            }, {
                output: [this.embeddingSize, contextSize],
                constants: {
                    outputSizes: splitW[i].length
                }
            })
            process.stdout.write(
                `\r[DEBUG] Weigths are Saved in VRAM!`
            );
        }
    }
    forward(input) {
        const result = Array.from({ length: this.contextSize }, () => new Float32Array(this.outputSize))
        let offset = 0;
        for (let i = 0; i < this.cacheLF.length; i++) {
            const out = this.cacheLF[i](this.Weights[i], this.Bias[i], input)
            for (let a = 0; a < this.contextSize; a++) {
                result[a].set(out[a], offset)
            }
            offset += out[0].length
        }
        return result
    }
    Save() {
        const weights = []
        const bias = []
        for (let a = 0; a < this.Weights.length; a++) {
            const chunkB = this.Bias[a]
            const chunkW = this.Weights[a]
            if (chunkB.toArray) {
                bias.push(...chunkB.toArray())
                weights.push(...chunkW.toArray())
            } else {
                bias.push(...chunkB)
                weights.push(...chunkW)
            }
        }
        return { weights, bias: new Float32Array(bias) }
    }
    backward(outGradient, input, learningRate) {
        const splitOut = split2DByColumns(outGradient)
        const result = Array.from({ length: this.contextSize }, () => new Float32Array(this.embeddingSize))
        for (let i = 0; i < splitOut.length; i++) {
            const oldW = this.Weights[i]
            const oldB = this.Bias[i]
            const out = this.cacheLInputGra[i](this.Weights[i], splitOut[i])
            this.Weights[i] = this.cacheLWeigths[i](this.Weights[i], learningRate, splitOut[i], input)
            this.Bias[i] = this.cacheLBias[i](this.Bias[i], learningRate, splitOut[i])
            for (let a = 0; a < this.contextSize; a++) {
                for (let g = 0; g < this.embeddingSize; g++) {
                    result[a][g] += out[a][g]
                }
            }
            if (oldW.delete) {
                oldW.delete()
                oldB.delete()
            }
        }
        return result
    }
}


export class GPUSoftmax {
    cacheSumCalculate;
    cacheMaxCache;
    cacheSF = [];
    cacheSB = []
    contextSize;
    calculateLoss;
    avgLoss;
    constructor(logits, contextSize) {
        this.contextSize = contextSize;
        this.avgLoss = gpu.createKernel(function (losses) {
            let sum = 0;
            for (let i = 0; i < this.constants.contextSize; i++) {
                sum += losses[i]
            }
            return sum / this.constants.contextSize
        }, {
            output: [1],
            constants: { contextSize }
        })
        this.calculateLoss = gpu.createKernel(function (probs, target) {
            return -Math.log(probs[this.thread.x][target[this.thread.x]] + 1e-9)
        }, { output: [contextSize], immutable: true, pipeline: true })
        this.cacheSumCalculate = gpu.createKernel(function (maxLogits, logits) {
            const maxLogit = maxLogits[this.thread.x]
            let sum = 0;
            for (let i = 0; i < this.constants.logitSize; i++) {
                sum += Math.exp(logits[this.thread.x][i] - maxLogit)
            }
            return sum
        }, {
            output: [contextSize],
            constants: { logitSize: logits[0].length },
            immutable: true,
            pipeline: true
        })
        this.cacheMaxCache = gpu.createKernel(function (logits) {
            let maxLogit = logits[this.thread.x][0]
            for (let i = 0; i < this.constants.logitSize; i++) {
                if (logits[this.thread.x][i] > maxLogit) {
                    maxLogit = logits[this.thread.x][i]
                }
            }
            return maxLogit
        }, {
            output: [contextSize],
            constants: { logitSize: logits[0].length },
            immutable: true,
            pipeline: true
        })
        const splited2d = split2DByColumns(logits)
        let offset = 0;
        for (let a = 0; a < splited2d.length; a++) {
            this.cacheSF[a] = gpu.createKernel(function (logits, sum, maxLogit) {
                return Math.exp(logits[this.thread.y][this.thread.x] - maxLogit[this.thread.y]) / sum[this.thread.y]
            }, {
                output: [splited2d[a][0].length, contextSize]
            })
            this.cacheSB[a] = gpu.createKernel(function (logits, targets) {
                let logit = logits[this.thread.y][this.thread.x]
                let globalIndex = this.constants.offset + this.thread.x
                return globalIndex == targets[this.thread.y] ? logit - 1.0 : logit
            }, {
                output: [splited2d[a][0].length, contextSize],
                constants: { offset: offset }
            })
            offset += splited2d[a][0].length
        }
    }
    forward(logits) {
        const maxLogits = this.cacheMaxCache(logits)
        const sumes = this.cacheSumCalculate(maxLogits, logits)
        const splitedLogits = split2DByColumns(logits)
        const result = Array.from({ length: this.contextSize }, () => [])
        for (let i = 0; i < splitedLogits.length; i++) {
            const out = this.cacheSF[i](splitedLogits[i], sumes, maxLogits)
            for (let a = 0; a < this.contextSize; a++) {
                result[a].push(...out[a])
            }
        }
        maxLogits.delete()
        sumes.delete()
        return result
    }
    backward(probs, target) {
        const splitedProbs = split2DByColumns(probs)
        const result = Array.from({ length: this.contextSize }, () => [])
        for (let i = 0; i < this.cacheSB.length; i++) {
            const out = this.cacheSB[i](splitedProbs[i], target)
            for (let a = 0; a < this.contextSize; a++) {
                result[a].push(...out[a])
            }
        }
        return result
    }
    crossEntropy(probs, target) {
        const losses = this.calculateLoss(probs, target)
        const loss = this.avgLoss(losses)
        losses.delete()
        return loss
    }

}


export class GPUHiddenLayer {
    cacheHF = [];
    cacheHB = []
    contextSize = 0;
    input = [];
    initKernel(probs, contextSize) {
        const splitL = split2DByColumns(probs)
        this.contextSize = contextSize;
        for (let a = 0; a < splitL.length; a++) {
            this.cacheHF[a] = gpu.createKernel(function (xs) {
                const x = xs[this.thread.y][this.thread.x]
                return x > 0 ? x : x * 0.02
            }).setOutput([splitL[a].length, contextSize])
            this.cacheHB[a] = gpu.createKernel(function (outputGradient, input) {
                const grad = outputGradient[this.thread.y][this.thread.x]
                const x = input[this.thread.y][this.thread.x]
                return x > 0 ? grad : grad * 0.02
            }, {
                output: [splitL[a].length, contextSize]
            })
        }
    }
    forward(logits) {
        if (this.cacheHF.length == 0) {
            this.initKernel(logits, logits.length)
        }
        this.input = logits
        const split = split2DByColumns(logits)
        const result = Array.from({ length: this.contextSize }, () => new Float32Array(logits[0].length))
        let offset = 0;
        for (let a = 0; a < split.length; a++) {
            const out = this.cacheHF[a](split[a])
            for (let i = 0; i < this.contextSize; i++) {
                result[i].set(out[i], offset)
            }
            offset += out[0].length
        }
        return result
    }
    backward(outputGradient) {
        const splitOut = split2DByColumns(outputGradient)
        const inputSp = split2DByColumns(this.input)
        const result = Array.from({ length: this.contextSize }, () => new Float32Array(outputGradient[0].length))
        let offset = 0;
        for (let a = 0; a < splitOut.length; a++) {
            const out = this.cacheHB[a](splitOut[a], inputSp[a])
            for (let i = 0; i < this.contextSize; i++) {
                result[i].set(out[i], offset)
            }
            offset += out[0].length
        }
        return result
    }
    ClearInputCache() {
        this.input = []
    }
}
export class GPUEmbedding {
    vocabSize = 0;
    embeddingSize = 0;
    contextSize = 0;
    chunkStarts = [];
    chunkEnds = [];
    vectors = [];

    lookupKernels = [];
    cacheEB = [];

    indexes = [];
    positions = [];

    constructor(vectors, vocabSize, embeddingSize, contextSize) {
        this.embeddingSize = embeddingSize;
        this.contextSize = contextSize;
        this.vocabSize = vocabSize;

        const splitedVectors = spilt2DArray(vectors);
        let start = 0;
        process.stdout.write(
            `\r[DEBUG] Saving Vectors in VRAM${".".repeat(Math.floor(Math.random() * 4))}`
        );
        for (let a = 0; a < splitedVectors.length; a++) {


            const length = splitedVectors[a].length;

            this.chunkStarts[a] = start;
            this.chunkEnds[a] = start + length;

            const vectorsK = gpu.createKernel(function (x) {
                return x[this.thread.x][this.thread.y];
            }, {
                output: [length, embeddingSize],
                pipeline: true,
                immutable: true
            });

            this.vectors[a] = vectorsK(splitedVectors[a]);
            vectorsK.destroy();

            this.cacheEB[a] = gpu.createKernel(function (vectors, indexes, positions, outGradient, learningRate) {
                const tokenIndex = this.thread.x;
                const dim = this.thread.y;
                let gradient = 0;

                for (let p = 0; p < this.constants.contextSize; p++) {
                    const index = indexes[p];
                    const pos = positions[p];

                    if (index >= 0 && index == tokenIndex && pos >= 0) {
                        gradient += outGradient[pos][dim];
                    }
                }

                return vectors[tokenIndex][dim] - learningRate * gradient;
            }, {
                output: [length, embeddingSize],
                immutable: true,
                pipeline: true,
                constants: {
                    contextSize: this.contextSize
                }
            });

            start += length;

        }
        process.stdout.write(
            `\n[DEBUG] Vectors are Saved in VRAM!`
        );

        process.stdout.write("\n");
    }

    getLookupKernel(count) {
        if (!this.lookupKernels[count]) {
            this.lookupKernels[count] = gpu.createKernel(function (vectors, indexes) {
                const dim = this.thread.x;
                const pos = this.thread.y;
                const index = Math.floor(indexes[pos]);

                return vectors[index][dim];
            }, {
                output: [
                    this.embeddingSize,
                    count
                ]
            });
        }

        return this.lookupKernels[count];
    }

    forward(tokens) {
        const result = Array.from(
            { length: this.contextSize },
            () => new Float32Array(this.embeddingSize)
        );

        for (let i = 0; i < this.vectors.length; i++) {

            const start = this.chunkStarts[i];
            const end = this.chunkEnds[i];

            const indexArray = new Float32Array(this.contextSize);
            const positionArray = new Int32Array(this.contextSize);

            for (let p = 0; p < this.contextSize; p++) {
                indexArray[p] = -1;
                positionArray[p] = -1;
            }

            let count = 0;

            for (let pos = 0; pos < tokens.length && count < this.contextSize; pos++) {
                const token = tokens[pos];
                if (token >= start && token < end) {
                    indexArray[count] = token - start;
                    positionArray[count] = pos;
                    count++;
                }
            }
            this.indexes[i] = indexArray;
            this.positions[i] = positionArray;
            if (count === 0) {
                continue;
            }
            const compactIndexes = indexArray.subarray(0, count);
            const lookup = this.getLookupKernel(count);
            const out = lookup(
                this.vectors[i],
                compactIndexes
            );

            for (let j = 0; j < count; j++) {
                const originalPosition = positionArray[j];
                for (let dim = 0; dim < this.embeddingSize; dim++) {
                    result[originalPosition][dim] = out[j][dim];
                }
            }
        }
        return result;
    }

    backward(outGradient, learningRate) {
        for (let a = 0; a < this.vectors.length; a++) {
            if (!this.indexes[a]) {
                continue;
            }

            let hasToken = false;

            for (let p = 0; p < this.contextSize; p++) {
                if (this.indexes[a][p] >= 0) {
                    hasToken = true;
                    break;
                }
            }

            if (!hasToken) {
                continue;
            }

            const oldVec = this.vectors[a];

            this.vectors[a] = this.cacheEB[a](
                oldVec,
                this.indexes[a],
                this.positions[a],
                outGradient,
                learningRate
            );

            if (oldVec && oldVec.delete) {
                oldVec.delete();
            }
        }
    }

    destroy() {
        for (const vector of this.vectors) {
            if (vector && vector.delete) {
                vector.delete();
            }
        }

        for (const kernel of this.lookupKernels) {
            if (kernel) {
                kernel.destroy();
            }
        }

        for (const kernel of this.cacheEB) {
            if (kernel) {
                kernel.destroy();
            }
        }

        this.vectors = [];
        this.lookupKernels = [];
        this.cacheEB = [];
        this.indexes = [];
        this.positions = [];
        this.chunkStarts = [];
        this.chunkEnds = [];
    }
}

export class GELUGPU {
    cacheGF;
    cacheGB;
    constructor(embeddingSize, contextSize) {
        this.cacheGF = gpu.createKernel(function (input) {
            const k = this.constants.kVal;
            const x = input[this.thread.y][this.thread.x]
            const u = k * (x + 0.044715 * x * x * x)
            return 0.5 * x * (1 + Math.tanh(u))
        }, {
            output: [embeddingSize, contextSize],
            constants: {
                kVal: Math.sqrt(2 / Math.PI)
            }
        })
        this.cacheGB = gpu.createKernel(function (input, gradient) {
            const x = input[this.thread.y][this.thread.x]
            const k = this.constants.kVal;
            const x3 = x * x * x;
            const u = k * (x + 0.044715 * x3)
            const t = Math.tanh(u)
            const sech2 = 1 - t * t
            const duDx = k * (1 + 3 * 0.044715 * x * x)
            const geluGrad =
                0.5 * (1 + t) +
                0.5 * x * sech2 * duDx;
            return gradient[this.thread.y][this.thread.x] * geluGrad
        }, {
            output: [embeddingSize, contextSize],
            constants: {
                kVal: Math.sqrt(2 / Math.PI)
            }
        })
    }
    forward(input) {
        return this.cacheGF(input)
    }
    backward(dOutput, input) {
        return this.cacheGB(input, dOutput)
    }
}