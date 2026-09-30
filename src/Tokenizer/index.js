import fs, { existsSync, readFileSync } from 'fs'
import { join, dirname, format } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url))
export class Tokenizer {
    vocabs = [];
    vocabSize = 0;
    IdtoWord = {};
    WordtoID = {};
    EOSToken = 2
    constructor(filename="") {
        if (fs.existsSync(filename)) {
            console.log("Warning: Tokenizer is loaded with a Custom Vocab File!")
            console.log("")
            this.vocabs = JSON.parse(readFileSync(filename, "utf-8"))
        } else {
            this.vocabs = JSON.parse(readFileSync(join(__dirname, "vocabs.json"), "utf-8"))
        }
        this.vocabSize = this.vocabs.length
        this.IdtoWord = Object.fromEntries(this.vocabs.map((e, idx) => [idx, e]))
        this.WordtoID = Object.fromEntries(this.vocabs.map((e, idx) => [e, idx]))
    }
    encoder(text = "") {
        const AddSpaces = text.toLocaleLowerCase().replaceAll(/([,.;'{}!`~\|":><?])/g, " $1 ").replaceAll("<" , " < ")
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
                    if (wordsliced.length == 0) output.push("<unk>")
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
        return output.filter(e => e.trim().length > 0).join(" ").replaceAll(/\s+([,.;'{}`!~\"?])/g, "$1")
    }
    trainTokenizer(text = "ejrnienieijr", mergeLength = 5, vocabFileName) {
        console.log("Training The Tokenizer....")
        const arrText = text.toLocaleLowerCase().replaceAll("\r\n", "").replaceAll("\n", "").replaceAll(/([,.;'{}!`~\|":><?])/g, " $1 ").split(/\s+/).map(e => e.trim()).filter((e) => e.length > 0)
        const reg = new RegExp(`.{0,${mergeLength}}`, "g")
        const prevVocab = existsSync(vocabFileName) ? JSON.parse(readFileSync(vocabFileName, "utf-8")) : [];
        const specal = ["<unk>",
            "<PAD>",
            "<EOS>",
            "$12",
            "$130",
            "😀",
            "😃",
            "😄",
            "😁",
            "😆",
            "😅",
            "😂",
            "🤣",
            "😊",
            "😎",
            "❤️",
            "👍",
            "👎",
            "🔥",
            "💯",
            "🎉",
            "🚀",
            "⭐",
            "😭",
            "😍",
            "🥰",
            "😡",
            "🤔",
            "😢",
            "😮",
            "👋",
            "🙏",
            "💀",
            "🤖",
            "👀",
            "💕",
            "💔",
            "😂😂",
            "😂😭",
            "🔥🔥",
            "❤️❤️",
            "👍👍",
            "<",
            ">",
            ".",
            "{",
            "}",
            "?",
            "!",
            "@",
            "$",
            ":",
            "/",
            "+",
            "-",
            "=",
            "%",
            "*",
            "@"
        ]
        const uniArr = [...new Set([...arrText, ...specal, ...prevVocab])];
        const output = []
        for(let a = 0 ; a < uniArr.length; a++){
            let text = uniArr[a].match(reg).filter((e)=> e.length > 0);
            output.push(...text)
        }
        let finalOutput = [...new Set([...output])]
        fs.writeFileSync(vocabFileName, JSON.stringify(finalOutput, null, 2), "utf-8");
        console.log("_________________________________")
        console.log(`New VocabSize : ${finalOutput.length - prevVocab.length}`);
        console.log("_________________________________")
        console.log("Operation Completed!")
        
    }
}