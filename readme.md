# TensorLoop ➰

A high-performance JavaScript library that provides essential AI and Large Language Model (LLM) development utilities. Build, train, and run neural network components like Embeddings, Linear layers, Activations, and Tokenizers directly in JavaScript.

---

## 🚀 Features

- **Transformer Block**: Complete modular block managing full attention and forward/backward training passes.
- **Embedding Vectors**: High-speed lookup and embedding initialization.
- **Linear Layer**: Fully connected layers with manageable weights and biases.
- **Activations**: Forward and backward propagation for **Softmax**, **ReLU**, and **GELU**.
- **Loss Functions**: **CrossEntropy** loss calculation for training classification tasks.
- **Attention & Positioning**: Built-in **Self-Attention** and **Positional Embedding** matrices.
- **Text Processing**: Built-in **Tokenizer** to convert raw text into token IDs and back.
- **Special Tokens**: Native support for `<PAD>` and `<EOS>`.
- **FFN BLOCK**: It add a layer to understand Complex Patterns.
---

## 🏗 Core Architecture

### Transformer Block
The `TransformerBlock` encapsulates attention, layer normalization, and feed-forward networks into a single trainable module.

```typescript
import { TransformerBlock } from '@mineproness/tensorloop'

// Initialize the block with an embedding size and save directory
const transformer = new TransformerBlock(1024, {
    dirname: "./model/transformer_layers"
})

// Forward Pass: Processes an array of typed float arrays
const output = transformer.forward(inputVectors)

// Backward Pass: Calculates gradients and updates weights using a learning rate
const dInput = transformer.backward(dOutput, 0.001)

// Clear cache to free up system memory after an iteration
transformer.clearInputCache()

// Persist the transformer block weights to disk
transformer.Save()
```

---

## 🛠 Configurations

You can configure saving mechanisms by passing a configuration object to the `Embedding` and `Linear` classes.

```javascript
import { Embedding, Linear } from '@mineproness/tensorloop'

// Initialize an Embedding layer and auto-save the weights
const embedding = new Embedding(1024, 2000, {
    save: {
        filename: "./model/vectors.bin"
    }
})

// Initialize a Linear layer (requires an array to save Weights and Biases separately)
const linear = new Linear(1024, 2000, {
    save: {
        filename: ["./model/Weights.bin", "./model/Bias.bin"]
    }
})
```

> 📌 **Note:** The `Linear` layer requires a two-element filename array to separate the Weights and Bias files. Always use the `.bin` extension.

---

## ⚡ Optimizations

- **Float32Array Powered**: Built on top of native typed arrays for maximum numerical performance and low memory overhead.
- **Heavy Workloads**: Easily handles large scaling operations, including dense 1024 embedding sizes.
- **Vanilla JavaScript**: Written entirely in raw JavaScript to eliminate bloated dependencies and optimize V8 execution.

---

## 🔤 Tokenizer

The tokenizer splits raw text strings into token IDs, which can then be fully decoded back into human-readable text.

### Example Usage:

```javascript
import { Tokenizer } from '@mineproness/tensorloop'

const tokenizer = new Tokenizer()
const text = "Hello, Welcome to My Channel."

// Encode text to token IDs
const { tokenIDs } = tokenizer.encoder(text) 
console.log(tokenIDs) // Output example: ["hello", "$12", "well", "come", "$130"...]

// Decode token IDs back to text
const decoded = tokenizer.decoder(tokenIDs) 
console.log(decoded) // Output: "Hello, Welcome to My Channel."
```

---

## 💡 Best Practices & Suggestions

### 1. Use TypeScript
We highly recommend using **TypeScript** with TensorLoop to enforce type safety and leverage IDE autocomplete features for all active classes and methods.

---

### 2. Always use NodeJS 20 Higher
This Package optimized for node js 20 or higher. If you use older version it Maybe Break Easliy.

---
### 3. Use GELU Always.
 
Gelu is Better Activation because it is not turn a gradient 0 so it is so smoothly works.

## ⭐ Support the Project

Thank you for visiting! If you find this library useful for your AI or LLM workflows, please consider giving us a star on **[GitHub](https://github.com/mineproness-org/tensorloop)**!
