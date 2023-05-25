import { dot, sigmoid } from "./Math"

export class Neuron{
    weights: number[]
    bias: number
    activation: number
    output: number

    constructor(inputCount: number){
        this.weights = new Array(inputCount).fill(0).map(x => Math.random())
        this.bias = Math.random()
        this.activation = 0
        this.output = 0
    }

    feedForward(inputs: number[]){
        this.activation = dot(this.weights, inputs) + this.bias
        this.output = sigmoid(this.activation)
    }
}