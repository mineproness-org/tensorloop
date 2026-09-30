import fs from 'fs'


export function getVector(filename, size, bias) {
    if (!fs.existsSync(filename)) {
        return null;
    }
    const bufferFloat32Array = fs.readFileSync(filename)
    const r = new Float32Array(bufferFloat32Array.buffer, bufferFloat32Array.byteOffset, bufferFloat32Array.byteLength / 4);
    if (r.length < size) {
         const output = new Float32Array(size);
         for(let a = 0; a < size; a++){
            let num = r[a];
            if(num){
                output[a] = num;
            }else{
                output[a] = bias ? 0 : (Math.random() * 2 - 1) * 0.02
            }
         }
    } else {
        return r
    }
}

export function SaveVector(filename, vectors) {
    const buffer = Buffer.from(vectors.buffer)
    fs.writeFileSync(filename, buffer)
}
