//https://victorzhou.com/blog/intro-to-neural-networks/

import { drawLittleGuy } from "./LittleGuy"
import { NeuralNetwork } from "./NeuralNetwork"
import { Neuron } from "./Neuron"


const canvas = document.getElementById("littleGuy") as HTMLCanvasElement
canvas.height = window.innerHeight
canvas.width = window.innerWidth
canvas.style.cursor = "none"
const pixel = ((window.innerHeight + window.innerWidth) / 2) / 400
const ctx = canvas.getContext("2d") as CanvasRenderingContext2D

drawLittleGuy(ctx, {x: canvas.width / 2, y: canvas.height / 2}, pixel)

canvas.addEventListener("pointermove", (e)=>{
  requestAnimationFrame(()=>{
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0,0,canvas.width, canvas.height)
      drawLittleGuy(ctx, {x: e.offsetX, y: e.offsetY}, pixel)
  })
})

const trainingInputs: number[][] = [
  [0,0], [1,0], [0,1], [1,1]
]

const trainingOutputs: number[][] = [
  [0], [1], [1], [0]
]



const testNetwork = new NeuralNetwork(trainingInputs[0].length, 20, trainingOutputs[0].length)

// console.log("initial output: " + testNetwork.outputNeurons.map(neuron=>neuron.output))

// testNetwork.feedForward(testData)

// console.log("first feedforward: " + testNetwork.outputNeurons.map(neuron=>neuron.output))

// testNetwork.backPropagate(testOutputs)

testNetwork.train(trainingInputs, trainingOutputs, 0.01, 200)

testNetwork.feedForward([0,0])
console.log(testNetwork.outputNeurons.map(neuron => neuron.output))


