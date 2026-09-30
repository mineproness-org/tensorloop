import { RELU, Linear } from "../../index.js";
import { join } from 'path'
import { OperationManager } from "../OperationManager.js";


export class FFN {
    linear1;
    linear2;
    hiddenLayer;
    Op
    constructor(device, embeddingSize, configs) {
        this.Op = new OperationManager(device)
        this.linear1 = new Linear(device, embeddingSize, embeddingSize * 4, {
            save: {
                filename: [join(configs.dirname, "FFN_weights1.bin"), join(configs.dirname, "FFN_bias1.bin")]
            }
        });
        this.hiddenLayer = new RELU(device, embeddingSize * 4);
        this.linear2 = new Linear(device, embeddingSize * 4, embeddingSize, {
            save: {
                filename: [join(configs.dirname, "FFN_weights2.bin"), join(configs.dirname, "FFN_bias2.bin")]
            }
        })
    }
    async forward(vec) {
        const logit1 = await this.linear1.forward(vec)
        const hidden = await this.hiddenLayer.forward(logit1)
        const final = await this.linear2.forward(hidden)
        return final
    }
    async backward(inputGradient, lr) {
        const logit2 = await this.linear2.backward(inputGradient, lr)
        const hidden = await this.hiddenLayer.backward(logit2)
        const final = await this.linear1.backward(hidden, lr)
        inputGradient.destroy()
        return final
    }
    async Save(){
        await this.linear1.Save()
        await this.linear2.Save()
    }
    async ClearInputCache(){
        this.linear1.ClearInputCache()
        this.linear2.ClearInputCache()
    }
}