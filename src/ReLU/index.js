import { GPUHiddenLayer } from "../gpiInit.js";

export class ReLU{
    input;
    gpu;
    constructor(){
        this.gpu = new GPUHiddenLayer()
    }
    forward(logits){
        return this.gpu.forward(logits)
    }
    backward(outGradient){
        return this.gpu.backward(outGradient)
    }
    
    ClearInputCache(){}
}