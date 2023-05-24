export const sigmoid = (x: number) =>{
    return 1 / (1 + Math.exp(-x))
}

export const d_sigmoid = (x: number) =>{
    const sx = sigmoid(x)
    return sx * (1 - sx)
}

export const sum = (x: number[]) =>{
    return x.reduce((previous, current) => previous + current)
}

export const dot = (x: number[], y: number[]) =>{
    return sum( x.map((a, i) => a * y[i]) )
}

export const scale = (vector: number[], scale: number) =>{
    return vector.map((x) => x * scale)
}