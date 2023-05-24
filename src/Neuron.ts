import { dot, sigmoid } from "./Math"

export class Neuron{
    weights: number[]
    bias: number

    constructor(inputCount: number){
        this.weights = new Array(inputCount).fill(0).map(x => Math.random())
        this.bias = Math.random()
    }

    feedForward(inputs: number[]){
        return sigmoid(dot(this.weights, inputs) + this.bias)
    }
}