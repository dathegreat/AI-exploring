import { d_sigmoid, sum, d_loss, hadamardProduct, scale, dot } from "./Math";
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
        console.log("Initialized")
        // console.log(this.layers[1][0].weights)
    }

    feedForward(inputs: number[]){
        // console.log("Feeding forward!")
        //feed forward initial inputs
        this.layers[0].map(neuron => neuron.feedForward(inputs))
        //feed forward through hidden layers
        for(let layer=1; layer<this.layers.length; layer++){
            const previousOutputs = this.layers[layer-1].map(neuron => neuron.output)
            this.layers[layer].map(neuron => neuron.feedForward(previousOutputs))
        }
        // for(const layer of this.layers){
        //     for( const neuron of layer){
        //         if(isNaN(neuron.output)){console.log("feedforward failed")}
        //     }
        // }
        // console.log(this.layers[1][0].weights)
    }
    //output neuron error calculated by [d_loss(neuron output) * d_sigmoid(neuron output)]
    //hidden neuron error calculated by [neuron weight * output neuron error * d_sigmoid(neuron output)]
    backPropagate(expectedOutputs: number[]){
        // console.log("Backpropagating!")
        //backprop output neurons
        for(let i=0; i<this.layers[this.layers.length - 1].length; i++){
            const neuron = this.layers[this.layers.length - 1][i]
            neuron.gradient = d_loss(neuron.output, expectedOutputs[i]) 
            // if(!neuron.gradient){console.log(
            //     `Gradient does not exist on neuron [${this.layers.length - 1}][${i}]
            //     weights   ${neuron.weights}
            //     activation ${neuron.activation}
            //     output ${neuron.output}`
            //     )}
            // if(neuron.gradient){console.log("output gradient exists " + neuron.gradient)}
        }
        //backprop hidden layer neurons
        for(let i=this.layers.length - 2; i>0; i--){
            for(let j=0; j<this.layers[i].length; j++){
                    const neuron = this.layers[i][j]
                    const previousLayerGradients = this.layers[i+1].map(neuron => neuron.gradient)
                    const previousLayerWeights = this.layers[i+1].map(neuron => neuron.weights[j])
                    const gradient = dot(previousLayerWeights, previousLayerGradients)
                    
                    neuron.gradient = gradient * d_sigmoid(neuron.output)
    //                 if(!neuron.gradient){console.log(`
    // Gradient does not exist on neuron [${i}][${j}]
    //     weights   ${neuron.weights}
    //     activation ${neuron.activation}
    //     output ${neuron.output}`
    //                     )}
    //                 if(neuron.gradient){console.log(`Gradient DOES exist on neuron [${i}][${j}]`)}
            }
            
        }
    //     console.log("Backprop completed!")
    //     console.log(this.layers[1][0].weights)
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
                        }
                    }
                }
                // console.log("Weights updated!")
                // console.log(this.layers.map(layer => layer.map(neuron =>neuron.weights)))
                // console.log(this.layers[1][0].weights)
                
            }
            if(epoch % 100 == 0){
                const errors = sum(this.layers.map(layer => sum(layer.map( neuron => neuron.gradient))))
                const totalError = Math.pow(errors, 2)
                console.log(`EPOCH ${epoch}`)
                console.log(`ERROR: ${totalError}`)
            }
        }
    }

    testInput(input: number[]){
        this.feedForward(input)
        return this.layers[this.layers.length-1].map(neuron => neuron.output)
    }

}
