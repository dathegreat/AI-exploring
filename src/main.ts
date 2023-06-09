//https://victorzhou.com/blog/intro-to-neural-networks/

import { drawLittleGuy } from "./LittleGuy"
import { NeuralNetwork } from "./NeuralNetwork"
import { Neuron } from "./Neuron"
import { Point } from "./Types"


const canvas = document.getElementById("littleGuy") as HTMLCanvasElement
canvas.height = window.innerHeight
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

const trainingInputs: number[][] = [
  [0,0], [1,0], [0,1], [1,1]
]

const trainingOutputs: number[][] = [
  [0,1], [1,0], [1,0], [0,1]
]



const testNetwork = new NeuralNetwork([trainingInputs[0].length, 4, 4, trainingOutputs[0].length])

const neuronLocations: Point[][] = new Array(testNetwork.layers.length).fill([]).map(element => [])
const scale = 150
const radius = scale / 2.5
for(let x=0; x<testNetwork.layers.length; x++){
  for(let y=0; y<testNetwork.layers[x].length; y++){
    neuronLocations[x][y] = {x: (x * scale) + radius, y: (y * scale) + canvas.width / 4}
  }
}

// testNetwork.layers.map((layer, x) => layer.map((neuron, y) => neuron.draw(ctx, neuronLocations[x][y], radius)))

testNetwork.train(trainingInputs, trainingOutputs, 0.1, 1000)

testNetwork.layers.map((layer, x) => layer.map((neuron, y) => neuron.draw(ctx, neuronLocations[x][y], radius)))

console.log(testNetwork.testInput([0,1]))
console.log(testNetwork.testInput([1,1]))
console.log(testNetwork.testInput([1,0]))
console.log(testNetwork.testInput([0,0]))
