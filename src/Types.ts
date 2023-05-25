export interface NeuronOutput{
    activation: number
    output: number
}

export interface FeedForwardOutput{
    hiddenLayer: NeuronOutput[]
    outputLayer: NeuronOutput[]
}