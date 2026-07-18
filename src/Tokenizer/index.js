import { readFileSync } from 'fs'
import { join, dirname, format } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url))
export class Tokenizer {
    vocabs = [];
    vocabSize = 0;
    IdtoWord = {};
    WordtoID = {};
    constructor() {
        this.vocabs = JSON.parse(readFileSync(join(__dirname, "vocabs.json"), "utf-8"))
        this.vocabSize = this.vocabs.length
        this.IdtoWord = Object.fromEntries(this.vocabs.map((e, idx) => [idx, e]))
        this.WordtoID = Object.fromEntries(this.vocabs.map((e, idx) => [e, idx]))
    }
    encoder(text = "") {
        const AddSpaces = text.toLocaleLowerCase().replaceAll(/([,.;'{}`~\|":><?])/g, " $1 ")
        const splitedWords = AddSpaces.split(/\s+/).filter(e => e.length > 0)
        const output = []
        for (let a = 0; a < splitedWords.length; a++) {
            const word = splitedWords[a]
            if (this.WordtoID[word]) {
                output.push(word)
            } else {
                let max = word.length
                output.push("$12")
                while (true) {
                    const regPatten = new RegExp(`.{0,${max}}`, "g")
                    const wordsliced = word.match(regPatten).filter((e) => e.length > 0)
                    const matched = wordsliced.flatMap(e => {
                        if (this.WordtoID[e]) {
                            return true
                        }
                        return false
                    
                    }).reduce((prev, curr) => prev && curr, true)
                    if(wordsliced.length == 0) console.error("Some Word Doesn't Exist in Vocabs List. so It can be undefined!")
                    if (matched) {
                        wordsliced.map((e) => output.push(e))
                        output.push("$130")
                        break
                    } else {
                        max -= 1
                        // break
                    }

                }
            }
        }
        return { tokens: output, tokenIDs: output.map((e) => this.WordtoID[e]) }
    }
    decoder(tokens = [1, 2, 3]) {
        const output = []
        let isSpace = false
        // let index = 0
        for (let token of tokens) {
            const formated = this.IdtoWord[String(token)]
            if (formated == "$12") {
                output[output.length] = ""
                isSpace = true
            } else if (formated == "$130") {
                isSpace = false
            } else {
                if (isSpace) {
                    output[output.length - 1] += formated
                } else {
                    output.push(formated)
                }
            }
        }
        return Validate(output).join(" ").replaceAll(/\s+([,.;'{}`~\|":><?])/g, "$1")
    }
}


function Validate(output = ["hello", "guys"]) {
    const bestOutput = []
    output.map((e, idx) => {
        if (idx == 0) {
            bestOutput[idx] = e[0].toLocaleUpperCase() + e.slice(1, e.length)
        } else {
            if (bestOutput[idx - 1] == ":" || bestOutput[idx - 1] == ".") {

                bestOutput[idx] = e[0].toLocaleUpperCase() + e.slice(1, e.length)
            } else {
                bestOutput[idx] = e

            }
        }

    })
    return bestOutput
}