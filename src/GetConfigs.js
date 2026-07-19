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

export function GetWeights(filename, embeddingSize) {
    const fileBuffer = fs.readFileSync(filename)
    const ELEMENTS_PER_ARRAY = embeddingSize;
    const BYTES_PER_ELEMENT = Float32Array.BYTES_PER_ELEMENT
    const BYTES_PER_ARRAY = ELEMENTS_PER_ARRAY * BYTES_PER_ELEMENT
    const totalArraySaved = fileBuffer.byteLength / BYTES_PER_ARRAY;
    const totalArray = []
    for (let i = 0; i < totalArraySaved; i++) {
        const currentOffset = fileBuffer.byteOffset + (i * BYTES_PER_ARRAY)
        const f32 = new Float32Array(fileBuffer.buffer, currentOffset, ELEMENTS_PER_ARRAY)
        totalArray.push(f32)
    }
    return totalArray
}

export function GetVector(filename) {
    const buffer = fs.readFileSync(filename)
    return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / Float32Array.BYTES_PER_ELEMENT)
}

export function Save(filename, vector) {
    return fs.writeFileSync(filename, Buffer.from(vector.buffer))
}

export function GetEmbeddingVectors(token, embeddingSize, filename) {
    const fd = fs.openSync(filename, "r")
    const PER_BTYES_VECTOR = embeddingSize * Float32Array.BYTES_PER_ELEMENT
    const buffer = Buffer.allocUnsafe(PER_BTYES_VECTOR)
    fs.readSync(fd, buffer, 0, PER_BTYES_VECTOR, token * PER_BTYES_VECTOR)
    fs.closeSync(fd)
    return new Float32Array(buffer)
}

export function SaveEmbeddingVectors(vectors = [new Float32Array([1, 2, 3])], filename) {
    const fd = fs.openSync(filename, "w")
    try {
        for (let a = 0; a < vectors.length; a++) {
            const vector = vectors[a]
            if (vector == undefined) continue;
            const offset = a * vector.length * Float32Array.BYTES_PER_ELEMENT;
            fs.readSync(fd, Buffer.from(vector.buffer) , 0 , vector.byteLength, offset)

        }
    }finally{
        fs.closeSync(fd)
    }
}