export class ReLU{
    input;
    forward(XLi){
        this.input = XLi
        return XLi.map((e)=> e > 0 ? e : 0)
    }
    backward(outputGradient){
        const inputGradient = new Float32Array(this.input.length)
        for(let i = 0; i < outputGradient.length; i++){
            inputGradient[i] = this.input[i] > 0 ? outputGradient[i] : 0
        }
        return inputGradient
    }
}