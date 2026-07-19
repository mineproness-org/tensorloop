import fs from "fs";
import { join } from "path";

export function SaveVectors(filename, vectors) {
    const fd = fs.openSync(filename, "w");

    try {
        for (const vector of vectors) {
            fs.writeSync(fd, Buffer.from(vector.buffer));
        }
    } finally {
        fs.closeSync(fd);
    }
}

export function GetVectors(filename, embeddingSize){
    const fileBuffer = fs.readFileSync(filename)
    const ELEMENTS_PER_ARRAY= embeddingSize;
    const BYTES_PER_ELEMENT= Float32Array.BYTES_PER_ELEMENT
    const BYTES_PER_ARRAY = ELEMENTS_PER_ARRAY * BYTES_PER_ELEMENT
    const totalArraySaved = fileBuffer.byteLength / BYTES_PER_ARRAY;
    const totalArray = []
    for(let i = 0; i < totalArraySaved; i++){
        const currentOffset = fileBuffer.byteOffset + (i * BYTES_PER_ARRAY)
        const f32 = new Float32Array(fileBuffer.buffer, currentOffset, ELEMENTS_PER_ARRAY)
        totalArray.push(f32)
    }
    return totalArray
}

export function GetVector(filename){
    const buffer = fs.readFileSync(filename)
    return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / Float32Array.BYTES_PER_ELEMENT)
}

export function Save(filename, vector){
    return fs.writeFileSync(filename, Buffer.from(vector.buffer))
}