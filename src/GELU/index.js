export class GELU {
    constructor() {
        this.inputs = []
    }

    forward(input, index = 0) {
        this.inputs[index] = new Float32Array(input)
        const output = new Float32Array(input.length)
        const k = Math.sqrt(2 / Math.PI)
        for (let i = 0; i < input.length; i++) {
            const x = input[i]
            const u = k * (x + 0.044715 * x * x * x)
            output[i] = 0.5 * x * (1 + Math.tanh(u))
        }
        return output
    }

    backward(dOutput, index = 0) {
        const input = this.inputs[index]
        if (!input) throw new Error(`No GELU cache for index ${index}`);
        const grad = new Float32Array(dOutput.length)
        const k = Math.sqrt(2 / Math.PI)
        for (let i = 0; i < dOutput.length; i++) {
            const x = input[i]
            const x3 = x * x * x
            const u = k * (x + 0.044715 * x3)
            const t = Math.tanh(u)
            const sech2 = 1 - t * t
            const duDx = k * (1 + 3 * 0.044715 * x * x)
            const geluGrad =
                0.5 * (1 + t) +
                0.5 * x * sech2 * duDx

            grad[i] = dOutput[i] * geluGrad
        }

        return grad
    }

    ClearInputCache() {
        this.inputs = []
    }
}