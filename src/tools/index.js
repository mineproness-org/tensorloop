export class tools{
    betterVector(vectors){
        return [...vectors].map((e)=> Number(e.toFixed(2)))
    }
}