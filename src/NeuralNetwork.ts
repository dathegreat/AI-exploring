import { d_sigmoid, sum, d_meanSquaredLoss, hadamardProduct, scale, dot, d_ReLU, shuffle, d_step, d_crossEntropyLoss, crossEntropyLoss } from "./Math";
import { Neuron } from "./Neuron";
import { TrainingData } from "./Types";

//https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
//https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L06%20Backpropagation.pdf
//https://cedar.buffalo.edu/~srihari/CSE676/6.5.2%20Chain%20Rule.pdf
//https://towardsdatascience.com/part-2-gradient-descent-and-backpropagation-bf90932c066a

const initializeNeurons = (amount: number, inputs: number, activationFunction: (x: number)=>number): Neuron[] =>{
    return new Array(amount).fill(0).map(()=>{
        return new Neuron(inputs, activationFunction)
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
    errors: number[]

    constructor(shape: number[], hiddenLayerActivation: (x: number)=>number, outputLayerActivation: (x: number)=>number){
        this.layers = new Array(shape.length)
        for(let i=0; i<shape.length; i++){
            if(i < shape.length - 1){
                this.layers[i] = initializeNeurons(shape[i], shape[i-1], hiddenLayerActivation)
            }else{
                this.layers[i] = initializeNeurons(shape[i], shape[i-1], outputLayerActivation)
            }
        }
        this.errors = []
        console.log("Initialized")
    }

    feedForward(inputs: number[]){
        //feed forward initial inputs
        this.layers[0].forEach(neuron => {neuron.feedForward(inputs)})
        //feed forward through hidden layers
        for(let layer=1; layer<this.layers.length; layer++){
            const previousOutputs = this.layers[layer-1].map(neuron => neuron.output)
            this.layers[layer].forEach(neuron => {neuron.feedForward(previousOutputs)})
        }
    }
    //output neuron error calculated by [d_loss(neuron output) * d_sigmoid(neuron output)]
    //hidden neuron error calculated by [neuron weight * output neuron error * d_sigmoid(neuron output)]
    backPropagate(expectedOutputs: number[]){
        //backprop output neurons
        for(let i=0; i<this.layers[this.layers.length - 1].length; i++){
            const neuron = this.layers[this.layers.length - 1][i]
            const outputs = this.layers[this.layers.length -1].map(neuron => neuron.output)
            neuron.gradient = d_crossEntropyLoss(outputs, expectedOutputs) * d_sigmoid(neuron.output)
            // console.log("backprop output")
            // neuron.print()
        }
        //backprop hidden layer neurons
        for(let i=this.layers.length - 2; i>=0; i--){
            for(let j=0; j<this.layers[i].length; j++){
                const neuron = this.layers[i][j]
                const previousLayerGradients = this.layers[i+1].map(neuron => neuron.gradient)
                const previousLayerWeights = this.layers[i+1].map(neuron => neuron.weights[j])
                neuron.gradient = dot(previousLayerWeights, previousLayerGradients) * d_ReLU(neuron.output)
                // console.log("backprop hidden")
                // neuron.print()
            }
        }
    }

    train(trainingData: TrainingData[], learningRate: number, momentum: number, epochs: number){
        for(let epoch=0; epoch<epochs; epoch++){
            const data = shuffle(trainingData)
            for(let input=0; input<trainingData.length; input++){
                this.feedForward(data[input].inputs)
                this.backPropagate(data[input].expected)
                //update weights
                for(let i=0; i<this.layers.length; i++){
                    const layer = this.layers[i]
                    for(let j=0; j<layer.length; j++){
                        const neuron = layer[j]
                        for(let k=0; k<neuron.weights.length; k++){
                            const neuronInput = i === 0 ? data[input].inputs[k] : this.layers[i-1][k].output
                            neuron.weights[k] = neuron.weights[k] - (learningRate * neuron.gradient * neuronInput) - (momentum * neuron.weights[k])
                            neuron.bias = neuron.bias - (learningRate * neuron.gradient)
                        }
                    }
                }
            }
            if (epoch % 10 === 0) {
                let totalError = 0;
                for(let input=0; input<trainingData.length; input++){
                    this.feedForward(data[input].inputs)
                    const predictedValues = this.layers[this.layers.length - 1].map(neuron => neuron.output)
                    const actualValues = data[input].expected
                    totalError += crossEntropyLoss(predictedValues, actualValues) / predictedValues.length
                }
                this.errors.push(totalError / trainingData.length)
                console.log("EPOCH: " + epoch + " ERROR: " + totalError / trainingData.length)
            }
        }
    }

    testInput(input: number[]){
        this.feedForward(input)
        return this.layers[this.layers.length-1].map(neuron => neuron.output)
    }

}
