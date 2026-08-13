export class GELU{
    input : Float32Array[];
    forward(input: Float32Array, index: number) : Float32Array;
    backward(Doutput: Float32Array,  index: number) : Float32Array;
    ClearInputCache() : void;
}