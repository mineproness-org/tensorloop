export class tools{
    betterVector(vectors){
        return JSON.stringify(vectors.map((e)=> e.toFixed(2)) , null , 2)
    }
}