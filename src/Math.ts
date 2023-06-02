export const sigmoid = (x: number) =>{
    return 1 / (1 + Math.exp(-x))
}

export const d_sigmoid = (x: number) =>{
    return x * (1 - x)
}

export const loss = (actual: number, expected: number) =>{
    return Math.pow(actual - expected, 2)
}

export const d_loss = (actual: number, expected: number) =>{
    return actual - expected
}

export const calculateNewWeight = (weight: number, learningRate: number, error: number) =>{
    return weight - (learningRate * error)
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