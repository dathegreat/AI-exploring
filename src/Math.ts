import { TrainingData } from "./Types"

export const step = (x: number) =>{
    return x >= 0 ? 1 : 0
}

export const d_step = (x: number) =>{
    return x
}

export const ReLU = (x: number) =>{
    return Math.max(0, x)
}

export const d_ReLU = (x: number) =>{
    return x < 0 ? 0 : 1
}

export const sigmoid = (x: number) =>{
    // if(isNaN(1 / (1 + Math.exp(-x)))){throw "absolutely sigmoid bonked. Input =  " + x}
    return 1 / (1 + Math.exp(-x))
}

export const d_sigmoid = (x: number) =>{
    // if(isNaN(x * (1 - x))){throw "absolutely d_sigmoid bonked. Input =  " + x}
    return x * (1 - x)
}

export const crossEntropyLoss = (predicted: number[], actual: number[]) =>{
    return sum(predicted.map((p, i) => ( -actual[i] * Math.log(p + Number.MIN_VALUE) ) - ( (1 - actual[i]) * Math.log(1 - p) )))
}

export const d_crossEntropyLoss = (predicted: number[], actual: number[]) =>{
    return sum(predicted.map((p, i) => (-actual[i] / p) + ((1 - actual[i])/(1 - p))))
}

export const meanSquaredLoss = (predicted: number[], actual: number[]) =>{
    return sum(predicted.map((p, i) => Math.pow(p - actual[i], 2)))
}

export const d_meanSquaredLoss = (predicted: number[], actual: number[]) =>{
    return sum(predicted.map((p, i) => 2 * (p - actual[i])))
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

export const truncate = (str: string, characterLimit: number) =>{
    return str.slice(0, characterLimit)
}

export const getRandomWeight = () =>{
    return -1 + (Math.random() * 2)
}

//Fisher-Yates shuffle algorithm
export const shuffle = (array: TrainingData[]) =>{
    let currentIndex = array.length,  randomIndex;
    // While there remain elements to shuffle.
    while (currentIndex != 0) {
    // Pick a remaining element.
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    // And swap it with the current element.
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
  }

  return array;
}