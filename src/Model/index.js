import { Embedding, Linear, Softmax, TransformerBlock, Tokenizer, PositionEmbedding } from '../../index.js'
import { join } from 'path'
import fs from 'fs'
import { OperationManager } from '../OperationManager.js'
export class Model {
    embedding;
    posEmbedding;
    linear;
    softmax;
    blocks = [];
    blockSize;
    dimSize;
    tokenizer;
    device;
    configs;
    tokenizerName;
    op;
    tokens;
    constructor(device, embeddingSize = 64, configs) {
        this.device = device;
        this.dimSize = embeddingSize;
        this.blockSize = configs.blockSize;
        this.configs = configs
        this.tokenizer = new Tokenizer(configs.tokenizerName)
        this.tokenizerName = configs.tokenizerName
        this.op = new OperationManager(device)
    }
    async CreateContext() {
        this.embedding = new Embedding(this.device, this.dimSize, this.tokenizer.vocabSize, {
            save: {
                filename: join(this.configs.dirname, "./vec.bin")
            }
        })
        this.linear = new Linear(this.device, this.dimSize, this.tokenizer.vocabSize, {
            save: {
                filename: [join(this.configs.dirname, "weights.bin"), join(this.configs.dirname, "bias.bin")]
            }
        })
        this.softmax = new Softmax(this.device, this.dimSize, this.tokenizer.vocabSize);
        for (let a = 0; a < this.blockSize; a++) {
            console.log(`Load the Transfomer Block ${a + 1}...`)
            if (!fs.existsSync(join(this.configs.dirname, `block-${a}`))) fs.mkdirSync(join(this.configs.dirname, `block-${a}`));
            this.blocks[a] = new TransformerBlock(this.device, this.dimSize, {
                dirname: join(this.configs.dirname, `block-${a}`)
            })
        }
        this.posEmbedding = new PositionEmbedding(this.device, this.dimSize, this.configs.contextSize, {
            save:{
                filename: join(this.configs.dirname, "posVec.bin")
            }
        })
    }
    TrainTokenizer(text = "", mergeSize = 5) {
        this.tokenizer.trainTokenizer(text, mergeSize, this.tokenizerName)
    }
    async ClearInputCache() {
        await this.linear.ClearInputCache()
        for (let a = 0; a < this.blockSize; a++) {
            await this.blocks[a].ClearInputCache()
        }
    }
    async forward(input, readBack) {
        if(this.configs.contextSize < input.length) throw new Error("The Input is larger then contextSize")
        this.tokens = input;
        await this.ClearInputCache()
        let vec = await this.embedding.forward(input)
        vec = await this.posEmbedding.forward(vec)
        for (let a = 0; a < this.blockSize; a++) {
            vec = await this.blocks[a].forward(vec)
        }
        let logits = await this.linear.forward(vec);
        let probs = await this.softmax.forward(logits, readBack)
        return probs
    }
    async backward(outputGradient, target, lr = 0.02) {
        const loss = await this.softmax.crossEntropy(outputGradient, target)
        const dLogits = await this.softmax.backward(outputGradient, target);
        let DInput = await this.linear.backward(dLogits, lr);
        for (let a = this.blockSize - 1; a !== 0; a--) {
            DInput = await this.blocks[a].backward(DInput, lr);
        }
        this.embedding.backward(this.tokens, DInput, lr);
        this.posEmbedding.backward(DInput, lr)
        return loss
    }
    async Save() {
        await this.embedding.Save()
        await this.linear.Save()
        for (let a = this.blockSize - 1; a !== 0; a--) {
            await this.blocks[a].Save();
        }
        await this.posEmbedding.Save()
    }
}