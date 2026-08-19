import { GPU } from "gpu.js";

const gpu = new GPU({
    mode: "gpu"
})
const kernel = gpu.createKernel(function () {
    return 1
}).setOutput([1])

kernel()
const gl = kernel.context

export function splitMatrix(matrix, chunksSizes = gl.getParameter(gl.MAX_TEXTURE_SIZE)) {
    const chunks = []
    for (let a = 0; a < matrix.length; a += chunksSizes) {
        chunks.push(matrix.slice(a, a + chunksSizes))
    }
    return { split: true, data: chunks }

}

const cacheLF = []
const cacheLBIas = []
const cacheLBWeights = []
const cacheLBInputGradent = []
export function InitKernel(weights, size) {
    const splitedWeights = splitMatrix(weights)
    for (let a = 0; a < splitedWeights.data.length; a++) {
        cacheLF[a] = gpu.createKernel(function (input, bias, weights) {
            let sum = bias[this.thread.x]
            for (let a = 0; a < this.constants.inputLength; a++) {
                sum += input[a] * weights[this.thread.x][a]
            }
            return sum
        }).setConstants({
            inputLength: splitedWeights.data[a].length
        }).setOutput([splitedWeights.data[a].length])
        cacheLBIas[a] = gpu.createKernel(function (
            bias,
            outputGradient,
            learningRate
        ) {
            const grad = Math.max(
                -1,
                Math.min(1, outputGradient[this.thread.x])
            )

            return bias[this.thread.x] - learningRate * grad
        })
            .setOutput([splitedWeights.data[a].length])
        cacheLBInputGradent[a] = gpu.createKernel(function (
            weights,
            outputGradient
        ) {
            let sum = 0

            for (let a = 0; a < this.constants.outputSize; a++) {
                const grad = Math.max(
                    -1,
                    Math.min(1, outputGradient[a])
                )

                sum += grad * weights[a][this.thread.x]
            }

            return sum
        })
            .setConstants({
                outputSize: splitedWeights.data[a].length
            })
            .setOutput([size])
        cacheLBWeights[a] = gpu.createKernel(function (
            weights,
            outputGradient,
            input,
            learningRate
        ) {
            const grad = Math.max(
                -1,
                Math.min(1, outputGradient[this.thread.y])
            )
          
            return weights[this.thread.y][this.thread.x]
                - learningRate * input[this.thread.x] * grad
        })
            .setOutput([size, splitedWeights.data[a].length])
    }


}
export function CalculateLinearForward(weights, bias, input) {
    const splitedBias = splitMatrix(bias)
    const splitedWeights = splitMatrix(weights)
    const result = new Float32Array(weights.length)
    let offset = 0;
    for (let a = 0; a < splitedBias.data.length; a++) {
        const output = cacheLF[a](input, splitedBias.data[a], splitedWeights.data[a])
        result.set(output, offset)
        offset += output.length
    }
    return result
}

export function CalculateBackward(weights, bias, out, learningRate, input) {
    const splitedBias = splitMatrix(bias)
    const splitedWeights = splitMatrix(weights)
    const inputGradient = new Float32Array(input.length)
    const biasOut = new Float32Array(bias.length)
    const weightsOut = []
    let offsetBias = 0
    let offsetWeights = 0
    let gradientOffset = 0

    for (let a = 0; a < splitedWeights.data.length; a++) {
        const weightChunk = splitedWeights.data[a]
        const biasChunk = splitedBias.data[a]

        const chunkOutputSize = weightChunk.length

        // Gradient belonging to this output/weight chunk
        const gradientChunk = out.slice(
            gradientOffset,
            gradientOffset + chunkOutputSize
        )

        gradientOffset += chunkOutputSize

        // Bias update
        const newBias = cacheLBIas[a](
            biasChunk,
            gradientChunk,
            learningRate
        )

        biasOut.set(newBias, offsetBias)
        offsetBias += newBias.length

        // Input gradient
        const outputGra = cacheLBInputGradent[a](
            weightChunk,
            gradientChunk
        )

        // ADD this chunk's gradient to the same input gradient
        for (let i = 0; i < outputGra.length; i++) {
            inputGradient[i] += outputGra[i]
        }

        // Weight update
        const newWeights = cacheLBWeights[a](
            weightChunk,
            gradientChunk,
            input,
            learningRate
        )

        // Copy chunk into final Float32Array
        weightsOut.push(...newWeights)
        
    }

    return {
        inputGradient,
        biasOut,
        weightsOut
    }
}