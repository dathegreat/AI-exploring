//https://victorzhou.com/blog/intro-to-neural-networks/

import { LearningCurve } from "./LearningCurve"
import { drawLittleGuy } from "./LittleGuy"
import { ReLU, shuffle, sigmoid, step } from "./Math"
import { NeuralNetwork } from "./NeuralNetwork"
import { Neuron } from "./Neuron"
import { Point, TrainingData } from "./Types"


const canvas = document.getElementById("littleGuy") as HTMLCanvasElement
canvas.height = window.innerHeight / 2
canvas.width = window.innerWidth
// canvas.style.cursor = "none"
const pixel = ((window.innerHeight + window.innerWidth) / 2) / 400
const ctx = canvas.getContext("2d") as CanvasRenderingContext2D

// drawLittleGuy(ctx, {x: canvas.width / 2, y: canvas.height / 2}, pixel)

// canvas.addEventListener("pointermove", (e)=>{
//   requestAnimationFrame(()=>{
//       ctx.setTransform(1, 0, 0, 1, 0, 0);
//       ctx.clearRect(0,0,canvas.width, canvas.height)
//       drawLittleGuy(ctx, {x: e.offsetX, y: e.offsetY}, pixel)
//   })
// })

// const unscaledTrainingInputs: number[][] = new Array(100).fill(1).map((x, index) => [index])

// const inputMin: number = Math.min(...unscaledTrainingInputs.flat())
// const inputMax: number = Math.max(...unscaledTrainingInputs.flat())

// const trainingInputs: number[][] = unscaledTrainingInputs.map((batch => batch.map(input => (input - inputMin) / (inputMax - inputMin))))

// const trainingOutputs: number[][] = unscaledTrainingInputs.map(x => [Math.pow(x[0], 2)])

const trainingInputs: number[][] = [
  [0, 0], [0, 1], [1, 0], [1, 1] 
]

const trainingOutputs: number[][] = [
  [0], [0], [0], [1]
]

// const trainingInputs: number[][] = new Array(1000).fill(1).map((x, index) => [index])

// const trainingOutputs: number[][] = trainingInputs.map(x => [x[0] + 1])

const trainingData: TrainingData[] = trainingInputs.map((input, index) => { return {inputs: input, expected: trainingOutputs[index]} } )

const testNetwork = new NeuralNetwork(
  [trainingInputs[0].length, 3, trainingOutputs[0].length],
  ReLU,
  sigmoid
)

const neuronLocations: Point[][] = new Array(testNetwork.layers.length).fill([]).map(element => [])
const scale = 150
const radius = scale / 2.5
for(let x=0; x<testNetwork.layers.length; x++){
  for(let y=0; y<testNetwork.layers[x].length; y++){
    neuronLocations[x][y] = {x: (x * scale) + radius, y: (y * scale) + canvas.width / 4}
  }
}

// testNetwork.layers.map((layer, x) => layer.map((neuron, y) => neuron.draw(ctx, neuronLocations[x][y], radius)))

testNetwork.train(trainingData, 0.03, 0.5, 1000)

testNetwork.layers.map((layer, x) => layer.map((neuron, y) => neuron.draw(ctx, neuronLocations[x][y], radius)))

console.log(testNetwork.layers[testNetwork.layers.length - 1])

console.log(testNetwork.testInput([0,0]))
console.log(testNetwork.testInput([0,1]))
console.log(testNetwork.testInput([1,0]))
console.log(testNetwork.testInput([1,1]))

// console.log(testNetwork.testInput([2]))

const chartCanvas = document.getElementById("learningCurve") as HTMLCanvasElement
const chartCtx = chartCanvas.getContext("2d") as CanvasRenderingContext2D
const learningChart = new LearningCurve(chartCtx, testNetwork.errors)