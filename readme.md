![tensor loop logo](https://i.imgur.com/I7WW3jf_d.webp?maxwidth=760&fidelity=grand)

A Library that Supply AI Tools and Large Language model Development tools like Embedding, Linear, ReLU, softmax etc...

## Feature

- Embedding Vectors.
- Linear Weights and Bias. 
- Softmax forward and Backward.
- CrossEntroy for Loss Calculation.
- Relu for Stable Layer.
- Tokenizer for convert into tokens.

## Configs

You can Pass a Object in a embedding and linear function. like :-

```js
import {Embedding , linear} from '@mineproness/tensorloop'

const embedding = new Embedding(1024, 2000, {
    save: {
        filename: "./model/vectors.bin"
    }
})
const linear = new Embedding(1024, 2000, {
    save: {
        filename: ["./model/Weights.bin" , "./model/Bias.bin"]
    }
})
```

Linear Layer need a filename array because it need to save the Weights and Bias and always use bin extension.

## Optimizations

It is Highly Fast Because of Float 32 Array and it is can handle 1024 embedding Size.
We build that with Raw Javascript for Best Optimization.

## Tokenizer

It is main use to Split Words and Convert into token ids that can be again Deconvert.


like this Exmple 

```javascript
import { Tokenizer } from '@mineproness/tensorloop'

const tokenizer = new Tokenizer()
const text = "Hello, Wellcome to My Channel."

const { tokenIDs } = tokenizer.encoder(text) // ["hello", "$12" , "well" , "come", "$130"....]
const decoded = tokenizer.decoder(tokenIDs) // Hello, wellcome to my channel.

```
## Suggestion

I recoommeded that use TypeScript for Type Safety and know what function is Exist in classes.

And Always Pair with Our tools in Our Package like
```js
import {Tools} from '@mineproness/tensorloop'

const tools = new Tools()

console.log(tools.betterVector(embeedingVectors))
```
## Thanks for Visiting this Site

If You watching this, Please give me a star in Github.