// const shuffle = (array: Array<any>) =>{
//   let currentIndex = array.length,  randomIndex;
//   // While there remain elements to shuffle.
//   while (currentIndex != 0) {
//     // Pick a remaining element.
//     randomIndex = Math.floor(Math.random() * currentIndex);
//     currentIndex--;
//     // And swap it with the current element.
//     [array[currentIndex], array[randomIndex]] = [
//       array[randomIndex], array[currentIndex]];
//   }
//   return array;
// }

import { sum } from "./Math";
import { Neuron } from "./Neuron";

const initializeNeurons = (amount: number, inputs: number): Neuron[] =>{
    return new Array(amount).fill(0).map(()=>{
        return new Neuron(inputs)
    })
}

const calculateLoss = (correct: number[], predicted: number[]) =>{
    const losses: number[] = []
    for(let i=0; i<correct.length; i++){
        losses[i] = Math.pow(correct[i] - predicted[i], 2)
    }
    return sum(losses) / losses.length
}

export class NeuralNetwork{
    hiddenNeurons: Neuron[]
    outputNeurons: Neuron[]

    constructor(inputLayerSize: number, hiddenLayerSize: number, outputLayerSize: number){
        this.hiddenNeurons = initializeNeurons(hiddenLayerSize, inputLayerSize)
        this.outputNeurons = initializeNeurons(outputLayerSize, hiddenLayerSize)
    }

    feedForward(inputs: number[]){
        //feed input into hidden neurons
        const hiddenOutput: number[] = this.hiddenNeurons.map(neuron => neuron.feedForward(inputs))
        const output: number[] = this.outputNeurons.map(neuron => neuron.feedForward(hiddenOutput))
        return output
    }

    backPropagate(){
        
    }

    train(learningRate: number, epochs: number){
        for(let i=0; i<epochs; i++){
            console.log(`EPOCH ${i}`)

        }
    }

}