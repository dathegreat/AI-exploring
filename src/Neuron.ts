import { dot, getRandomWeight, sigmoid, truncate } from "./Math"
import { Point } from "./Types"

export class Neuron{
    weights: number[]
    bias: number
    activationFunction: (x: number)=>number
    activation: number
    output: number
    gradient: number

    constructor(inputCount: number, activationFunction: (x: number)=>number){
        this.weights = new Array(inputCount).fill(0).map(x => getRandomWeight())
        this.bias = getRandomWeight()
        this.activationFunction = activationFunction
        this.activation = 0
        this.output = 0
        this.gradient = 0
    }

    feedForward(inputs: number[]){
        this.activation = dot(this.weights, inputs) + this.bias
        this.output = this.activationFunction(this.activation)
    }

    draw(ctx: CanvasRenderingContext2D, center: Point, radius: number){
        const outputString = truncate(this.output.toString(), 4)
        const weightsString = this.weights.reduce((str, weight) => str + truncate(weight.toString(), 4) + " | ", "| ")
        const biasString = truncate(this.bias.toString(), 4)
        const gradientString = truncate(this.gradient.toString(), 5)
        ctx.beginPath()
        ctx.arc(center.x, center.y, radius, 0, Math.PI * 2)
        ctx.stroke()
        ctx.closePath()
        ctx.font = "bold 8pt sans-serif"
        ctx.textAlign = "center"
        ctx.fillText("Weights: " + weightsString, center.x, center.y - (ctx.measureText(weightsString).actualBoundingBoxAscent) * 2)
        ctx.fillText("Bias: " + biasString, center.x, center.y)
        ctx.fillText("Output: " + outputString, center.x, center.y + (ctx.measureText(biasString).actualBoundingBoxAscent) * 2)
        ctx.fillText("Gradient: " + gradientString, center.x, center.y + (ctx.measureText(outputString).actualBoundingBoxAscent) * 4)
    }

    print(){
        console.log(
            `Weights: ${this.weights}\nBias: ${this.bias}\nActivation: ${this.activation}\nOutput: ${this.output}\nGradient: ${this.gradient}`
        )
    }
}