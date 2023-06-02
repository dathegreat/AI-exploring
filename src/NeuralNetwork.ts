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

//https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

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
        //feed input to hidden neurons
        this.hiddenNeurons.map(neuron => neuron.feedForward(inputs))
        //feed hidden output to output neurons
        this.outputNeurons.map(neuron => neuron.feedForward(this.hiddenNeurons.map(hidden => hidden.output)))
    }
    //output neuron error calculated by [d_loss(neuron output) * d_sigmoid(neuron output)]
    //hidden neuron error calculated by [neuron weight * output neuron error * d_sigmoid(neuron output)]
    backPropagate(expectedOutputs: number[]){
        this.outputNeurons.map(
            (neuron, index) => {
                neuron.errors = [d_loss(neuron.output, expectedOutputs[index]) * d_sigmoid(neuron.output)]
            }
        )
        this.hiddenNeurons.map(
            (neuron, index) => {
                neuron.errors = neuron.weights.map(
                    weight => sum( this.outputNeurons.map( outputNeuron => (weight * outputNeuron.errors[0] * d_sigmoid(neuron.output)) ))
                )
            }
        )
    }

    train(inputs: number[][], expectedOutputs: number[][], learningRate: number, epochs: number){
        for(let i=0; i<epochs; i++){
            for(let j=0; j<inputs.length; j++){
                this.feedForward(inputs[j])
                this.backPropagate(expectedOutputs[j])
                //update weights
                //NOTE TO FUTURE D.A. this is what is breaking the code
                for(const neuron of this.hiddenNeurons){
                    neuron.weights = neuron.weights.map(
                        (weight, index) => {
                            return (weight - learningRate) * neuron.errors[index] * inputs[j][index]
                        }) 
                }
                for(const neuron of this.outputNeurons){
                    neuron.weights = neuron.weights.map(
                        (weight, index) => {
                            return (weight - learningRate) * neuron.errors[index] * this.hiddenNeurons[index].output
                        })
                }
            }
            const totalError = Math.pow(sum(this.hiddenNeurons.map(neuron => sum(neuron.errors))) + sum(this.outputNeurons.map(neuron => sum(neuron.errors))), 2)
            if(i % 100 == 0){
                console.log(`EPOCH ${i}`)
                console.log(`ERROR: ${totalError}`)
            }
        }
    }

}
