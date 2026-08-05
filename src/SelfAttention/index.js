import { Softmax } from '../../index.js';
import { Linear } from '../Linear/index.js'

export class SelfAttention {
    query;
    key;
    value;
    v;
    k;
    q;
    input;
    configs;
    score;
    softmax;
    weights;
    constructor(embeddingSize, configs) {
        this.configs = configs
        this.softmax = new Softmax()
        this.score = Math.sqrt(embeddingSize)
        this.query = new Linear(embeddingSize, embeddingSize, { save: { filename: [`./${configs.dirname}/query.bin`, `./${configs.dirname}/queryBias.bin`] } })
        this.key = new Linear(embeddingSize, embeddingSize, { save: { filename: [`./${configs.dirname}/key.bin`, `./${configs.dirname}/keyBias.bin`] } })
        this.value = new Linear(embeddingSize, embeddingSize, { save: { filename: [`./${configs.dirname}/value.bin`, `./${configs.dirname}/valueBias.bin`] } })
    }
    forward(vectors) {
        this.q = vectors.map((e) => this.query.forward(e))
        this.v = vectors.map((e) => this.value.forward(e))
        this.k = vectors.map((e) => this.key.forward(e))
        // console.log(this.q, this.k, this.v)
        const attentionSocres = []
        for (let i = 0; i < this.q.length; i++) {
            attentionSocres[i] = new Float32Array(this.k.length)
            for (let a = 0; a < this.k.length; a++) {
                let dot = 0;
                for (let d = 0; d < this.q[i].length; d++) {
                    dot += this.q[i][d] * this.k[a][d]
                }
                attentionSocres[i][a] = dot / this.score
            }
        }
        const weights = attentionSocres.map((e) => this.softmax.forward(e))
        this.weights = weights
        const output = [];

        for (let i = 0; i < weights.length; i++) {
            output[i] = new Float32Array(this.v[0].length);

            for (let j = 0; j < this.v.length; j++) {
                for (let d = 0; d < this.v[j].length; d++) {
                    output[i][d] += weights[i][j] * this.v[j][d];
                }
            }
        }

        return output;
    }
    softmaxBack(probs, dOut) {
        const dIn = new Float32Array(probs.length);

        for (let i = 0; i < probs.length; i++) {
            let sum = 0;

            for (let j = 0; j < probs.length; j++) {
                const jacobian =
                    i === j
                        ? probs[i] * (1 - probs[i])
                        : -probs[i] * probs[j];

                sum += jacobian * dOut[j];
            }

            dIn[i] = sum;
        }

        return dIn;
    }
    backward(dOutput, learningRate) {
        const dV = this.v.map((e) => new Float32Array(e.length));
        const dWeights = []
        for (let a = 0; a < this.weights.length; a++) {
            dWeights[a] = new Float32Array(this.weights[a].length)
            for (let j = 0; j < this.v.length; j++) {
                for (let r = 0; r < this.v[j].length; r++) {
                    dWeights[a][j] += dOutput[a][r] * this.v[j][r]
                    dV[j][r] += dOutput[a][r] * this.weights[a][j]
                }
            }
        }
        const dScores = []
        for (let t = 0; t < dWeights.length; t++) {
            dScores[t] = this.softmaxBack(this.weights[t], dWeights[t])
        }
        const dQ = this.q.map(q => new Float32Array(q.length));
        const dK = this.k.map(k => new Float32Array(k.length));
        for (let a = 0; a < this.q.length; a++) {
            for (let j = 0; j < this.k.length; j++) {
                for (let r = 0; r < this.q[a].length; r++) {
                    dQ[a][r] += dScores[a][j] * this.k[j][r] / this.score;
                    dK[j][r] += dScores[a][j] * this.q[a][r] / this.score;
                }
            }
        }
        const dInput = this.q.map(() => new Float32Array(this.q[0].length))
        for (let a = 0; a < this.q.length; a++) {
            const dq = this.query.backward(dQ[a], learningRate, a)
            const dk = this.key.backward(dK[a], learningRate, a)
            const dv = this.value.backward(dV[a], learningRate, a)
            for (let b = 0; b < dq.length; b++) {
                dInput[a][b] = dq[b] + dk[b] + dv[b]
            }
        }
        return dInput
    }
    Save() {
        this.query.Save()
        this.key.Save()
        this.value.Save()
    }
}