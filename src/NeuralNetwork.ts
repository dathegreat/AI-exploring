import { d_sigmoid, sum, d_loss, calculateNewWeight } from "./Math";
import { Neuron } from "./Neuron";

//https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

const initializeNeurons = (amount: number, inputs: number): Neuron[] =>{
    return new Array(amount).fill(0).map(()=>{
        return new Neuron(inputs)
    })
}

const checkNaN = (n: number) =>{
    if(isNaN(n)){
        throw "NOT A NUMBER";
    }
    return n
}

export class NeuralNetwork{
    layers: Neuron[][]

    constructor(shape: number[]){
        this.layers = new Array(shape.length)
        for(let i=1; i<shape.length; i++){
            this.layers[i] = initializeNeurons(shape[i], shape[i-1])
        }
        console.log(this.layers)
    }

    feedForward(inputs: number[]){
        //feed forward initial inputs
        this.layers[0].map(neuron => neuron.feedForward(inputs))
        //feed forward through hidden layers
        for(let layer=1; layer<this.layers.length; layer++){
            this.layers[layer].map(neuron => neuron.feedForward(this.layers[layer-1].map(previous => previous.output)))
        }
    }
    //output neuron error calculated by [d_loss(neuron output) * d_sigmoid(neuron output)]
    //hidden neuron error calculated by [neuron weight * output neuron error * d_sigmoid(neuron output)]
    backPropagate(expectedOutputs: number[]){
        for(let i=0; i<this.outputNeurons.length; i++){
            const neuron = this.outputNeurons[i]
            neuron.errors = [d_loss(neuron.output, expectedOutputs[i]) * d_sigmoid(neuron.output)]
            this.outputNeurons[i] = neuron
        }
        for(let i=this.hiddenNeurons.length-1; i>0; i--){
            for(let j=0; j<this.hiddenNeurons[i].length; j++){
                const neuron = this.hiddenNeurons[i][j]
                neuron.errors[j] = neuron.weights[j] * this.hiddenNeurons[i+1][j].errors[j] * d_sigmoid(neuron.output)  
            }
        }
    }

    train(inputs: number[][], expectedOutputs: number[][], learningRate: number, epochs: number){
        for(let epoch=0; epoch<epochs; epoch++){
            for(let input=0; input<inputs.length; input++){
                this.feedForward(inputs[input])
                this.backPropagate(expectedOutputs[input])
                //update weights
                for(const neuron of this.outputNeurons){
                    for(let i=0; i<neuron.weights.length; i++){
                        neuron.weights[i] = neuron.weights[i] - (learningRate * neuron.errors[i] * this.hiddenNeurons[this.hiddenNeurons.length - 1][i].output)
                    }
                }
                
            }
            if(epoch % 100 == 0){
                const outputError = sum(this.outputNeurons.map(neuron => sum(neuron.errors)))
                const hiddenError = sum(this.hiddenNeurons.map(layer => sum(layer.map( neuron => neuron.output))))
                const totalError = Math.pow(hiddenError + outputError, 2)
                console.log(`EPOCH ${epoch}`)
                console.log(`ERROR: ${totalError}`)
            }
        }
    }

}
