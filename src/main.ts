//https://victorzhou.com/blog/intro-to-neural-networks/

import { NeuralNetwork } from "./NeuralNetwork"
import { Neuron } from "./Neuron"

const trainingInputs: number[][] = [
  [0,0], [1,0], [0,1], [1,1]
]

const trainingOutputs: number[][] = [
  [0], [1], [1], [0]
]

const canvas = document.getElementById("littleGuy") as HTMLCanvasElement
canvas.height = window.innerHeight
canvas.width = window.innerWidth
canvas.style.cursor = "none"
const pixel = ((window.innerHeight + window.innerWidth) / 2) / 400
const ctx = canvas.getContext("2d") as CanvasRenderingContext2D

const drawLittleGuy = (center: {x: number, y: number}) =>{
  ctx.translate(center.x, center.y)
  const legTheta = Math.PI / 3
  const bodyRadius = pixel * 2
  const leftLegStart = {
    x: -(Math.cos(legTheta) * bodyRadius), 
    y: (Math.sin(legTheta) * bodyRadius)
  }
  const rightLegStart = {
    x: (Math.cos(legTheta) * bodyRadius), 
    y: (Math.sin(legTheta) * bodyRadius)
  }
  //draw body
  ctx.beginPath()
  ctx.arc(0, 0, pixel * 2, 0, Math.PI * 2)
  ctx.stroke()
  ctx.closePath()
  //draw legs
  ctx.beginPath()
  ctx.moveTo(leftLegStart.x, leftLegStart.y)
  ctx.lineTo(leftLegStart.x * 1.5, leftLegStart.y * 1.5)
  ctx.moveTo(rightLegStart.x, rightLegStart.y)
  ctx.lineTo(rightLegStart.x * 1.5, rightLegStart.y * 1.5)
  ctx.stroke()
  ctx.closePath()
  //draw eyes
  ctx.beginPath()
  ctx.arc(pixel, -pixel / 2, pixel / 4, 0, Math.PI * 2)
  ctx.stroke()
  ctx.beginPath()
  ctx.arc(-pixel, -pixel / 2, pixel / 4, 0, Math.PI * 2)
  ctx.stroke()
  //draw mouth
  ctx.beginPath()
  ctx.moveTo(-pixel / 2, pixel / 4)
  ctx.bezierCurveTo(-pixel, pixel, pixel, pixel, pixel / 2, pixel / 4)
  ctx.stroke()
  //draw hat
  ctx.translate(0, -bodyRadius)
  ctx.moveTo(0,0)
  ctx.lineTo(-bodyRadius, 0)
  ctx.lineTo(-bodyRadius, -pixel)
  ctx.lineTo(-bodyRadius / 2, -pixel)
  ctx.lineTo(-bodyRadius / 2, -pixel * 2.5)
  ctx.lineTo(bodyRadius / 2, -pixel * 2.5)
  ctx.lineTo(bodyRadius / 2, -pixel)
  ctx.lineTo(bodyRadius, -pixel)
  ctx.lineTo(bodyRadius, 0)
  ctx.lineTo(0,0)
  ctx.stroke()
}

drawLittleGuy({x: canvas.width / 2, y: canvas.height / 2})

canvas.addEventListener("pointermove", (e)=>{
  requestAnimationFrame(()=>{
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0,0,canvas.width, canvas.height)
      drawLittleGuy({x: e.offsetX, y: e.offsetY})
  })
})


const testData = [0,1,1,2,0,3,4,5]
const testNetwork = new NeuralNetwork(testData.length, 20, 1)
console.log(testNetwork.feedForward(testData))

