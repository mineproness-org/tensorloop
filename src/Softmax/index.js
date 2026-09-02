import { GPUSoftmax } from "../gpiInit.js";


export class Softmax{
    gpu;
    forward(xLi){
        if(!this.gpu){
            this.gpu = new GPUSoftmax(xLi, xLi.length)
        }
        return this.gpu.forward(xLi)
    }
    backward(xLi , target){
        return this.gpu.backward(xLi, target)
    }
    crossEntropy(probs, target){
        return this.gpu.crossEntropy(probs, target)
    }
}