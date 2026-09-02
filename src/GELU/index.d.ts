export class GELU{
    input : Float32Array[];
    forward(input: Float32Array[]) : Float32Array[];
    backward(Doutput: Float32Array[]) : Float32Array[];
    ClearInputCache() : void;
}