import { d_sigmoid, sum, d_loss, hadamardProduct, scale, dot } from "./Math";
import { Neuron } from "./Neuron";

//https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
//https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L06%20Backpropagation.pdf
//https://cedar.buffalo.edu/~srihari/CSE676/6.5.2%20Chain%20Rule.pdf
//https://towardsdatascience.com/part-2-gradient-descent-and-backpropagation-bf90932c066a

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
        console.log("Initialized")
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
            neuron.gradient = d_loss(neuron.output, expectedOutputs[i]) * d_sigmoid(neuron.output)
        }
        //backprop hidden layer neurons
        for(let i=this.layers.length - 2; i>0; i--){
            for(let j=0; j<this.layers[i].length; j++){
                    const neuron = this.layers[i][j]
                    const previousLayerGradients = this.layers[i+1].map(neuron => neuron.gradient)
                    const previousLayerWeights = this.layers[i+1].map(neuron => neuron.weights[j])
                    neuron.gradient = dot(previousLayerWeights, previousLayerGradients) * d_sigmoid(neuron.output)
            }
        }
    }

    train(inputs: number[][], expectedOutputs: number[][], learningRate: number, epochs: number){
        for(let epoch=0; epoch<epochs; epoch++){
            for(let input=0; input<inputs.length; input++){
                this.feedForward(inputs[input])
                this.backPropagate(expectedOutputs[input])
                //update weights
                for(let i=0; i<this.layers.length; i++){
                    for(let j=0; j<this.layers[i].length; j++){
                        for(let k=0; k<this.layers[i][j].weights.length; k++){
                            const previousLayerOutput = i > 0 ? this.layers[i-1][k].output : inputs[i][j]
                            this.layers[i][j].weights[k] = this.layers[i][j].weights[k] - (learningRate * this.layers[i][j].gradient * previousLayerOutput)
                            this.layers[i][j].bias = this.layers[i][j].bias - (learningRate * this.layers[i][j].gradient)
                        }
                    }
                }
            }
            if(epoch % 10 == 0){
                const errors = sum(this.layers.map(layer => sum(layer.map( neuron => neuron.gradient))))
                const totalError = Math.pow(errors, 2)
                console.log(`EPOCH ${epoch}`)
                console.log(`LOSS: ${totalError}`)
            }
        }
    }

    testInput(input: number[]){
        this.feedForward(input)
        return this.layers[this.layers.length-1].map(neuron => neuron.output)
    }

}
