export class Softmax{
    forward(xLi : number[]) : number
    backward(target : number[]) : Float32Array[]
    crossEntropy(dLogits: Float32Array[], target: number[])
}