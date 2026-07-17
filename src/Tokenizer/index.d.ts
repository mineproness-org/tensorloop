export class Tokenizer{
    vocabs : string[]
    vocabSize: number
    IdtoWord : Record<string, string>
    WordtoID : Record<string , number>
    encoder(text: string) : {
        tokens: string[]
        tokenIDs: number[]
    }
    decoder(tokens: number[] | number[][]) : string
}