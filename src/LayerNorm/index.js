export class LayerNorm {
    gamma;
    beta;
    eps;
    cache;

    constructor(embeddingSize, eps = 1e-5) {
        this.gamma = new Float32Array(embeddingSize);
        this.beta = new Float32Array(embeddingSize);

        this.gamma.fill(1);
        this.beta.fill(0);

        this.eps = eps;
        this.cache = [];
    }

    forward(input, index = 0) {
        const size = input.length;
        let mean = 0;
        for (let i = 0; i < size; i++) {
            mean += input[i];
        }
        mean /= size;
        let variance = 0;
        for (let i = 0; i < size; i++) {
            const diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= size;
        const invStd = 1 / Math.sqrt(variance + this.eps);
        const output = new Float32Array(size);
        for (let i = 0; i < size; i++) {
            const normalized = (input[i] - mean) * invStd;
            output[i] = normalized * this.gamma[i] + this.beta[i];
        }
        this.cache[index] = {
            input: new Float32Array(input),
            mean,
            invStd
        };
        return output;
    }

    backward(dOutput, learningRate, index = 0) {
        const cache = this.cache[index];
        const input = cache.input;
        const mean = cache.mean;
        const invStd = cache.invStd;
        const size = input.length;
        const dInput = new Float32Array(size);
        let sumDNorm = 0;
        let sumDNormX = 0;
        const normalized = new Float32Array(size);
        for (let i = 0; i < size; i++) {
            normalized[i] = (input[i] - mean) * invStd;
            const dNorm = dOutput[i] * this.gamma[i];
            sumDNorm += dNorm;
            sumDNormX += dNorm * normalized[i];
            const dGamma = dOutput[i] * normalized[i];
            const dBeta = dOutput[i];

            this.gamma[i] -= learningRate * dGamma;
            this.beta[i] -= learningRate * dBeta;
        }
        for (let i = 0; i < size; i++) {
            const dNorm = dOutput[i] * this.gamma[i];
            dInput[i] =
                (invStd / size) *
                (
                    size * dNorm -
                    sumDNorm -
                    normalized[i] * sumDNormX
                );
        }
        return dInput;
    }

    ClearInputCache() {
        this.cache = [];
    }
}