export class Softmax{
    forward(xLi){
        const max = Math.max(...xLi)
        const exp = xLi.map((e)=> Math.exp(e - max))
        const sum = exp.reduce((a,b)=> a + b , 0)
        return exp.map((e)=> e / sum)
    }
    backward(probs , target){
        const pro = [...probs]
        pro[target] -= 1
        return pro
    }
}