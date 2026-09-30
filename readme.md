# Tensorloop

**GPU-accelerated neural network and Transformer library for JavaScript using WebGPU.**

Tensorloop is a JavaScript library for building and training neural networks and GPT-style Transformer models directly with **WebGPU**.

It is designed around GPU computation while keeping the API relatively simple, making it possible to experiment with embeddings, linear layers, self-attention, Transformer blocks, softmax, and training without relying on Python-based ML frameworks.

> 🚧 **Status:** Experimental / actively developing

---

## ✨ Features

* ⚡ WebGPU-accelerated computation
* 🧠 Transformer architecture components
* 🔤 Token embeddings
* 🔢 Linear layers
* 🎯 Softmax and cross-entropy
* 🔄 Forward and backward propagation
* 🧩 Transformer blocks
* 💾 Model weight saving
* 📦 JavaScript / Node.js
* 🖥️ GPU computation through WebGPU
* 🛠️ Designed for building LLM experiments from scratch

---

## 🧠 Transformer Architecture

Tensorloop can be used to construct a GPT-style language model using individual components.

A typical model can look like:

```text
                    Token IDs
                       │
                       ▼
                  ┌──────────┐
                  │ Tokenizer│
                  └────┬─────┘
                       │
                       ▼
                  ┌──────────┐
                  │ Embedding│
                  └────┬─────┘
                       │
                       ▼
              ┌─────────────────┐
              │ TransformerBlock│
              │  Self-Attention │
              │      + FFN      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ TransformerBlock│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ TransformerBlock│
              └────────┬────────┘
                       │
                       ▼
                  ┌──────────┐
                  │  Linear  │
                  └────┬─────┘
                       │
                       ▼
                  ┌──────────┐
                  │ Softmax  │
                  └────┬─────┘
                       │
                       ▼
                 Token Probabilities
```

The components can be connected manually, allowing you to experiment with the architecture instead of using a single black-box model.

---

## 📦 Installation

Install Tensorloop with npm:

```bash
npm install @mineproness/tensorloop
```

Tensorloop requires a runtime with WebGPU support.

For Node.js WebGPU environments, you can use a WebGPU implementation such as:

```bash
npm install webgpu
```

---

## 🚀 Basic Example

The following example creates a small language model using:

* Tokenizer
* 64-dimensional embeddings
* 3 Transformer blocks
* Linear output layer
* Softmax
* Cross-entropy loss
* Backpropagation
* GPU computation

```js
import {
    Embedding,
    Linear,
    Tokenizer,
    TransformerBlock,
    Softmax
} from "@mineproness/tensorloop"

import { create, globals } from "webgpu"
import fs from "fs"

Object.assign(globalThis, globals)

const navigator = {
    gpu: create(["adapter=NVIDIA GeForce 920M"])
}

const contextSize = 24
const lr = 0.02

async function train() {

    // Load tokenizer
    const tokenizer = new Tokenizer("./vocab.json")

    // Initialize WebGPU
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "low-power"
    })

    const device = await adapter.requestDevice()

    // Token embedding
    const embedding = new Embedding(
        device,
        64,
        tokenizer.vocabSize,
        {
            cpuReadBack: false,

            save: {
                filename: "model/vec.bin"
            }
        }
    )

    // Output projection
    const linear = new Linear(
        device,
        64,
        tokenizer.vocabSize,
        {
            save: {
                filename: [
                    "./model/bias.bin",
                    "./model/weights.bin"
                ]
            }
        }
    )

    // Transformer blocks
    const block = new TransformerBlock(
        device,
        64,
        {
            dirname: "./model/block1"
        }
    )

    const block1 = new TransformerBlock(
        device,
        64,
        {
            dirname: "./model/block2"
        }
    )

    const block2 = new TransformerBlock(
        device,
        64,
        {
            dirname: "./model/block3"
        }
    )

    // Output softmax
    const softmax = new Softmax(
        device,
        64,
        tokenizer.vocabSize
    )

    // Load dataset
    const tokens = fs
        .readFileSync("./dataset.txt", "utf-8")
        .split("\r\n")
        .filter(e => e.length > 0)
        .map(e => e.trim() + "<EOS>")
        .map(e => tokenizer.encoder(e).tokenIDs)
        .flat()

    // Create next-token prediction pairs
    const trainingPairs = []

    for (let i = 0; i < tokens.length; i++) {

        trainingPairs.push({
            inputIDX: tokens.slice(
                i,
                i + contextSize
            ),

            targetIDX: tokens.slice(
                i + 1,
                i + contextSize + 1
            )
        })

    }

    const start = 0
    const end = 8000

    // Training
    for (let epoch = 0; epoch < 100; epoch++) {

        let totalLoss = 0
        let totalAcc = 0

        for (let i = start; i < end; i++) {

            // -------------------------
            // Forward pass
            // -------------------------

            let vec = await embedding.forward(
                trainingPairs[i].inputIDX
            )

            vec = await block.forward(vec)

            vec = await block1.forward(vec)

            vec = await block2.forward(vec)

            const logits = await linear.forward(vec)

            const probs = await softmax.forward(logits)

            // -------------------------
            // Loss
            // -------------------------

            const { loss, acc } =
                await softmax.crossEntropy(
                    probs,
                    trainingPairs[i].targetIDX
                )

            totalLoss += loss
            totalAcc += acc

            // -------------------------
            // Backward pass
            // -------------------------

            const dLogits =
                await softmax.backward(
                    probs,
                    trainingPairs[i].targetIDX
                )

            let dInput =
                await linear.backward(
                    dLogits,
                    lr
                )

            dInput =
                await block2.backward(
                    dInput,
                    lr
                )

            dInput =
                await block1.backward(
                    dInput,
                    lr
                )

            dInput =
                await block.backward(
                    dInput,
                    lr
                )

            embedding.backward(
                trainingPairs[i].inputIDX,
                dInput,
                lr
            )

            // Clear cached tensors
            await block.ClearInputCache()
            await block1.ClearInputCache()
            await block2.ClearInputCache()
            await linear.ClearInputCache()
        }

        // -------------------------
        // Save model
        // -------------------------

        await embedding.Save()
        await linear.Save()
        await block.Save()
        await block1.Save()
        await block2.Save()

        console.log(
            `Epoch ${epoch + 1} | ` +
            `Loss: ${totalLoss / (end - start)} | ` +
            `Accuracy: ${totalAcc / (end - start)}`
        )
    }
}

train()
```

---

# 🔤 Tokenization

Tensorloop can work with a custom tokenizer.

For example:

```js
const tokenizer = new Tokenizer("./vocab.json")
```

Encode text into token IDs:

```js
const result = tokenizer.encoder(
    "Hello world"
)

console.log(result.tokenIDs)
```

The resulting token IDs can then be passed directly into the embedding layer:

```js
const vectors = await embedding.forward(
    result.tokenIDs
)
```

This makes it possible to use your own tokenizer and vocabulary instead of requiring a predefined tokenizer.

---

# 🧩 Embedding

The `Embedding` layer converts token IDs into dense vectors.

Example:

```js
const embedding = new Embedding(
    device,
    64,
    tokenizer.vocabSize
)
```

Here:

```text
64
```

is the embedding dimension.

For a vocabulary of `V` tokens, the embedding matrix contains:

```text
V × 64
```

parameters.

Forward pass:

```js
const vectors = await embedding.forward(tokens)
```

---

# 🧱 TransformerBlock

Transformer blocks combine the main components required for Transformer-based models.

```js
const block = new TransformerBlock(
    device,
    64,
    {
        dirname: "./model/block1"
    }
)
```

Multiple blocks can be stacked:

```js
vec = await block.forward(vec)

vec = await block1.forward(vec)

vec = await block2.forward(vec)
```

And during training:

```js
dInput = await block2.backward(
    dInput,
    lr
)

dInput = await block1.backward(
    dInput,
    lr
)

dInput = await block.backward(
    dInput,
    lr
)
```

This allows Tensorloop to build deeper Transformer networks by stacking blocks.

---

# 🎯 Linear Layer

The `Linear` layer projects the Transformer representation into vocabulary space.

```js
const linear = new Linear(
    device,
    64,
    tokenizer.vocabSize
)
```

For example:

```text
Embedding size = 64
Vocabulary = 3,000

64 → 3,000
```

The output contains one logit for every vocabulary token.

---

# 📊 Softmax

The `Softmax` layer converts logits into probabilities.

```js
const probs = await softmax.forward(logits)
```

The predicted probability distribution can then be evaluated using cross-entropy:

```js
const { loss, acc } =
    await softmax.crossEntropy(
        probs,
        targetTokens
    )
```

---

# 🔄 Backpropagation

Tensorloop supports manually controlled backpropagation.

The gradients flow backwards through the network:

```text
Softmax
   ↓
Linear
   ↓
TransformerBlock 3
   ↓
TransformerBlock 2
   ↓
TransformerBlock 1
   ↓
Embedding
```

In code:

```js
const dLogits =
    await softmax.backward(
        probs,
        targetIDX
    )

let dInput =
    await linear.backward(
        dLogits,
        lr
    )

dInput =
    await block2.backward(
        dInput,
        lr
    )

dInput =
    await block1.backward(
        dInput,
        lr
    )

dInput =
    await block.backward(
        dInput,
        lr
    )

embedding.backward(
    inputIDX,
    dInput,
    lr
)
```

This gives you direct control over the training process.

---

# ⚡ WebGPU

Tensorloop is designed around WebGPU.

Example:

```js
import { create, globals } from "webgpu"

Object.assign(globalThis, globals)

const navigator = {
    gpu: create([
        "adapter=NVIDIA GeForce 920M"
    ])
}
```

Then request the GPU device:

```js
const adapter =
    await navigator.gpu.requestAdapter()

const device =
    await adapter.requestDevice()
```

The device is passed to Tensorloop layers:

```js
const embedding =
    new Embedding(device, 64, vocabSize)

const linear =
    new Linear(device, 64, vocabSize)

const block =
    new TransformerBlock(device, 64)
```

This lets the layers perform their GPU computations through WebGPU.

---



# 🔤 Training the Tokenizer

Tensorloop includes a tokenizer that can learn a vocabulary directly from your dataset.

```js
import fs from "fs"
import { Tokenizer } from "./index.js"

const tokenizer = new Tokenizer("./vocab.json")

tokenizer.trainTokenizer(
    fs.readFileSync("./dataset.txt", "utf-8"),
    6,
    "./vocab.json"
)
```

## `trainTokenizer()`

```js
tokenizer.trainTokenizer(
    text,
    maxMergeLength,
    outputPath
)
```

| Parameter        | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `text`           | Training text used to build the tokenizer vocabulary |
| `maxMergeLength` | Maximum length used when creating token merges       |
| `outputPath`     | Location where the generated vocabulary is saved     |

### Example

```js
tokenizer.trainTokenizer(
    fs.readFileSync("./dataset.txt", "utf-8"),
    6,
    "./vocab.json"
)
```

Here:

```text
6
```

is the **maximum merge length**.

The tokenizer learns merges from the dataset and saves the resulting vocabulary to:

```text
vocab.json
```

### Tokenizer Workflow

```text
                 dataset.txt
                      │
                      ▼
              ┌─────────────────┐
              │ trainTokenizer()│
              │                 │
              │ maxMergeLength=6│
              └────────┬────────┘
                       │
                       ▼
                   vocab.json
                       │
                       ▼
                ┌────────────┐
                │  Tokenizer │
                └─────┬──────┘
                      │
                      ▼
              tokenizer.encoder()
                      │
                      ▼
                  Token IDs
```

Once the vocabulary has been created, load it with:

```js
const tokenizer = new Tokenizer("./vocab.json")
```

Then encode text:

```js
const result = tokenizer.encoder(
    "Hello world"
)

console.log(result.tokenIDs)
```

The resulting token IDs can be passed directly into Tensorloop's embedding layer:

```js
const vectors = await embedding.forward(
    result.tokenIDs
)
```

This allows you to create a tokenizer vocabulary from your own dataset and then use that vocabulary for Transformer training.


---




# 💾 Saving Models

Tensorloop layers can save their parameters to files.

Embedding:

```js
const embedding = new Embedding(
    device,
    64,
    vocabSize,
    {
        save: {
            filename: "model/vec.bin"
        }
    }
)
```

Linear:

```js
const linear = new Linear(
    device,
    64,
    vocabSize,
    {
        save: {
            filename: [
                "./model/bias.bin",
                "./model/weights.bin"
            ]
        }
    }
)
```

Transformer blocks can use their own directories:

```js
const block = new TransformerBlock(
    device,
    64,
    {
        dirname: "./model/block1"
    }
)
```

A model can therefore be organized like:

```text
model/
├── vec.bin
├── weights.bin
├── bias.bin
│
├── block1/
│   ├── ...
│
├── block2/
│   ├── ...
│
└── block3/
    ├── ...
```

---

# 🧪 Example Model Configuration

The example above uses:

| Component           |               Configuration |
| ------------------- | --------------------------: |
| Embedding dimension |                          64 |
| Context length      |                          24 |
| Transformer blocks  |                           3 |
| Vocabulary          |       `tokenizer.vocabSize` |
| Learning rate       |                        0.02 |
| Training pairs      |                       8,000 |
| Epochs              |                         100 |
| Optimizer           | SGD-style parameter updates |

The vocabulary size is determined dynamically from the tokenizer:

```js
tokenizer.vocabSize
```

So the same architecture can be used with different vocabulary sizes.

---

# 📁 Example Project

A simple project can look like:

```text
my-model/
│
├── dataset.txt
├── vocab.json
├── train.js
│
└── model/
    ├── vec.bin
    ├── weights.bin
    ├── bias.bin
    │
    ├── block1/
    ├── block2/
    └── block3/
```

Run training with:

```bash
node train.js
```

---

# 🛠️ Design Philosophy

Tensorloop is intended to make the internals of neural networks accessible from JavaScript.

Instead of hiding everything behind a single high-level training function, Tensorloop exposes the individual operations:

```js
embedding.forward()
transformer.forward()
linear.forward()
softmax.forward()

softmax.backward()
linear.backward()
transformer.backward()
embedding.backward()
```

This makes it useful for:

* Learning how neural networks work
* Experimenting with Transformer architectures
* Building small language models
* WebGPU experiments
* GPU programming experiments
* Creating custom neural-network layers
* Understanding forward and backward propagation

---

# 🧠 Building an LLM From Scratch

Tensorloop can be used as a foundation for experimenting with GPT-style language models.

A simplified training pipeline is:

```text
Dataset
   │
   ▼
Tokenizer
   │
   ▼
Token IDs
   │
   ▼
Embedding
   │
   ▼
Transformer Blocks
   │
   ▼
Linear Projection
   │
   ▼
Softmax
   │
   ▼
Cross Entropy
   │
   ▼
Backpropagation
   │
   └──────────────► Update weights
```

The architecture can then be expanded with additional Transformer blocks, larger embeddings, larger vocabularies, and other normalization or optimization techniques.

---

# 🔬 Experimental

Tensorloop is still an experimental project.

The API and internal implementation may change as new features are added.

The project is primarily intended for experimentation, learning, and building custom GPU-accelerated neural-network systems.

---

# 📜 License



MIT License


---

## 👨‍💻 Author

Created by **mineproness**.

Tensorloop is part of the broader **@mineproness** ecosystem of JavaScript AI/ML experiments.

---

# ⭐ If you find Tensorloop interesting

Give the project a star ⭐ and experiment with building your own neural networks and Transformer models directly in JavaScript.
