export class ReLU{
    forward(XLi : Float32Array[] | number[]) : number[] | Float32Array[];
    backward(outputGradient: number[] | Float32Array[]) : Float32Array[];
}