export const sigmoid = (x: number) =>{
    if(isNaN(1 / (1 + Math.exp(-x)))){throw "absolutely sigmoid bonked. Input =  " + x}
    return 1 / (1 + Math.exp(-x))
}

export const d_sigmoid = (x: number) =>{
    if(isNaN(x * (1 - x))){throw "absolutely d_sigmoid bonked. Input =  " + x}
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
//computes the element-wise product of two vectors
export const hadamardProduct = (a: number[], b: number[]) =>{
    return a.map((a_i, i) => a_i * b[i])
}