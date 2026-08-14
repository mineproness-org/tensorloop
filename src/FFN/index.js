import { Linear } from '../Linear/index.js'
import { GELU } from '../GELU/index.js'
import { ReLU } from '../ReLU/index.js'
import { join } from 'path'
export class FFN {
    linear1;
    linear2;
    hidden;
    constructor(embeddingSize, hiddenLayer = "gelu", configs) {
        this.hidden = hiddenLayer.toLocaleLowerCase() == "gelu" ? new GELU() : new ReLU()
        if (configs) {
            this.linear1 = new Linear(embeddingSize, 4 * embeddingSize, {
                save: {
                    filename: [join(configs.dirname, "ffnWeights.bin"), join(configs.dirname, "ffnBias.bin")]
                }
            })
            this.linear2 = new Linear(4 * embeddingSize, embeddingSize, {
                save: {
                    filename: [join(configs.dirname, "ffnWeights1.bin"), join(configs.dirname, "ffnBias1.bin")]
                }
            })
        } else {
            this.linear1 = new Linear(embeddingSize, 4 * embeddingSize)
            this.linear2 = new Linear(4 * embeddingSize, embeddingSize)
        }
    }
    forward(input){
        const result = []
        for(let a = 0; a < input.length; a++){
            const out1 = this.linear1.forward(input[a])
            const hidden = this.hidden.forward(out1, a)
            result[a] = this.linear2.forward(hidden)
        }
        return result
    }
    backward(dInput, LearningRate){
        const result = []
        for(let a = 0; a < dInput.length; a++){
           const out2 = this.linear2.backward(dInput[a] , LearningRate, a);
           const dHidden = this.hidden.backward(out2, a)
           result[a] = this.linear1.backward(dHidden, LearningRate, a)
        }
        return result
    }
    Save(){
        this.linear1.Save()
        this.linear2.Save()
    }
    ClearInputCache(){
        this.linear1.ClearInputCache()
        this.linear2.ClearInputCache()
        this.hidden.ClearInputCache()
    }
}