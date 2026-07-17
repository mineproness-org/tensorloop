export function crossEntropy(probs, target){
    return -Math.log(probs[target] + 1e-10)
}
