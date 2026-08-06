import fs from 'fs'
import { join } from 'path'

export function SaveVectors(vectors, path){
    const rows = vectors.length;
    const cols = vectors[0].length;
    const flatArray = new Float32Array(rows * cols)
    vectors.forEach((e,i)=>{
        flatArray.set(e, i * cols)
    })
    const bufferArray = Buffer.from(flatArray.buffer, flatArray.byteOffset, flatArray.byteLength)
    fs.writeFileSync(path, bufferArray)
}

export function LoadVectors(path, cols){
   const buffer = fs.readFileSync(path)
   const flatArray = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4 )
   const vectors = []
   const rows = flatArray.length / cols;
   for(let a = 0; a < rows; a++){
       vectors.push(flatArray.slice(a * cols, (a+1) * cols))
   }
   return vectors
}

export function SaveBias(vectors, path){
    const bufferArray = Buffer.from(vectors.buffer, vectors.byteOffset, vectors.byteLength)
    fs.writeFileSync(path, bufferArray)
}

export function LoadBias(path){
   const buffer = fs.readFileSync(path)
   const flatArray = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4 )
   return flatArray
}