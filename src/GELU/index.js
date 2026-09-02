import { GELUGPU } from "../gpiInit.js";

export class GELU {
    gpu;
    inputs = [];
    constructor() {
        this.inputs = []
        
    }

    forward(input) {
        this.inputs = input
        if(!this.gpu){
            this.gpu = new GELUGPU(input[0].length, input.length)
        }
        // const output = new Float32Array(input.length)
        // const k = Math.sqrt(2 / Math.PI) 
        // for (let i = 0; i < input.length; i++) {
        //     const x = input[i]
        //     const u = k * (x + 0.044715 * x * x * x)
        //     output[i] = 0.5 * x * (1 + Math.tanh(u))
        // }
        return this.gpu.forward(input)
    }

    backward(dOutput) {  
        return this.gpu.backward(dOutput, this.inputs)
    }

    ClearInputCache() {
        this.inputs = []
    }
}