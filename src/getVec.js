import fs from 'fs'


export function getVector(filename, size, bias) {
    if (!fs.existsSync(filename)) {
        return null;
    }
    const buffer = fs.readFileSync(filename);
    const r = new Float32Array(
        buffer.buffer,
        buffer.byteOffset,
        buffer.byteLength / 4
    );
    if (r.length >= size) {
        return r;
    }
    const output = new Float32Array(size);
    output.set(r);
    if (!bias) {
        for (let i = r.length; i < size; i++) {
            output[i] = (Math.random() * 2 - 1) * 0.02;
        }
    }
    return output;
}

export function SaveVector(filename, vectors) {
    const buffer = Buffer.from(vectors.buffer)
    fs.writeFileSync(filename, buffer)
}
