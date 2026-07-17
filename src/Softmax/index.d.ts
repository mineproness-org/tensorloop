export class Softmax{
    forward(xLi : number[]) : number[]
    backward(probalitils: number[] , target : number) : number[] | Float32Array[]
}