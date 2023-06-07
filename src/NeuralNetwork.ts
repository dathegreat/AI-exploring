import { d_sigmoid, sum, d_loss, calculateNewWeight, hadamardProduct, scale } from "./Math";
import { Neuron } from "./Neuron";

//https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
//https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L06%20Backpropagation.pdf
//https://cedar.buffalo.edu/~srihari/CSE676/6.5.2%20Chain%20Rule.pdf
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
        for(let i=0; i<shape.length; i++){
            this.layers[i] = initializeNeurons(shape[i], shape[i-1])
        }
        console.log(this.layers)
    }

    feedForward(inputs: number[]){
        //feed forward initial inputs
        this.layers[0].map(neuron => neuron.feedForward(inputs))
        //feed forward through hidden layers
        for(let layer=1; layer<this.layers.length; layer++){
            const previousOutputs = this.layers[layer-1].map(neuron => neuron.output)
            this.layers[layer].map(neuron => neuron.feedForward(previousOutputs))
        }
    }
    //output neuron error calculated by [d_loss(neuron output) * d_sigmoid(neuron output)]
    //hidden neuron error calculated by [neuron weight * output neuron error * d_sigmoid(neuron output)]
    backPropagate(expectedOutputs: number[]){
        //backprop output neurons
        for(let i=0; i<this.layers[this.layers.length - 1].length; i++){
            const neuron = this.layers[this.layers.length - 1][i]
            const gradients = scale(neuron.weights, d_loss(neuron.output, expectedOutputs[i]))
            neuron.errors =  scale(gradients, d_sigmoid(neuron.output))
            if(neuron.errors.length != neuron.weights.length){console.log("NOOOOOO")}
        }
        //backprop hidden layer neurons
        for(let i=this.layers.length - 2; i>0; i--){
            for(let j=0; j<this.layers[i+1].length; j++){
                for(let k=0; k<this.layers[i][j].weights.length; k++){
                    const neuron = this.layers[i][j]
                    const gradients = hadamardProduct(this.layers[i+1][j].errors, this.layers[i+1][j].weights)
                    if(gradients){console.log(this.layers[i+1][j].errors.length, this.layers[i+1][j].weights.length, i, j)}
                    neuron.errors[k] = sum(gradients) * d_sigmoid(neuron.output) 
                    
                }
            }
        }
    }

    train(inputs: number[][], expectedOutputs: number[][], learningRate: number, epochs: number){
        for(let epoch=0; epoch<epochs; epoch++){
            for(let input=0; input<inputs.length; input++){
                this.feedForward(inputs[input])
                this.backPropagate(expectedOutputs[input])
                //update weights
                // for(const neuron of this.outputNeurons){
                //     for(let i=0; i<neuron.weights.length; i++){
                //         neuron.weights[i] = neuron.weights[i] - (learningRate * neuron.errors[i] * this.hiddenNeurons[this.hiddenNeurons.length - 1][i].output)
                //     }
                // }
                
            }
            if(epoch % 100 == 0){
                const outputError = sum(this.layers[this.layers.length-1].map(neuron => sum(neuron.errors)))
                const hiddenError = sum(this.layers.map(layer => sum(layer.map( neuron => neuron.output))))
                const totalError = Math.pow(hiddenError + outputError, 2)
                console.log(`EPOCH ${epoch}`)
                console.log(`ERROR: ${totalError}`)
            }
        }
    }

}
