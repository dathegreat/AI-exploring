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

import { d_sigmoid, sum, d_loss, calculateNewWeight } from "./Math";
import { Neuron } from "./Neuron";
import { FeedForwardOutput, NeuronOutput } from "./Types";

const initializeNeurons = (amount: number, inputs: number): Neuron[] =>{
    return new Array(amount).fill(0).map(()=>{
        return new Neuron(inputs)
    })
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
        this.hiddenNeurons.map(neuron => neuron.feedForward(inputs))
        //feed hidden neurons' output into output neurons
        this.outputNeurons.map(neuron => neuron.feedForward(this.hiddenNeurons.map(hidden => hidden.output)))
    }
    //output neuron error calculated by [d_loss(neuron output) * d_sigmoid(neuron output)]
    //hidden neuron error calculated by [neuron weight * output neuron error * d_sigmoid(neuron output)]
    //NOTE TO FUTURE D.A.: this math could totally be wrong, please verify
    backPropagate(expectedOutputs: number[]){
        const outputLayerErrors: number[] = this.outputNeurons.map(
            (neuron, index) => d_loss(expectedOutputs[index], neuron.output) * d_sigmoid(neuron.output)
        )
        const hiddenLayerErrors: number[] = this.hiddenNeurons.map(
            (neuron, index) => (sum(neuron.weights) / neuron.weights.length) * outputLayerErrors[index] * d_sigmoid(neuron.output)
        )
        
    }

    train(inputs: number[], expectedOutputs: number[], learningRate: number, epochs: number){
        for(let i=0; i<epochs; i++){
            console.log(`EPOCH ${i}`)
            const outputs = this.feedForward(inputs)
            const costs = this.backPropagate(expectedOutputs, outputs)
            
        }
    }

}
