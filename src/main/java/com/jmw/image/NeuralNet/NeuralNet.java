package com.jmw.image.NeuralNet;

/**
 * Fully Connected Neural Network
 * 
 * @author Jimmy Whitaker
 */
public class NeuralNet
{ 
	private int numLayers;
	// TODO Add Mask for DropConnect
	public Layer[] layers;

	/**
	 * Constructs a new neural network given the parameters.
	 * 
	 * @param layerNeurons Array of the number of neurons per layer
	 * @param layerType Array of Activation Functions used for each layer
	 * @param inputNeurons number of inputs to the neural network
	 */
	public NeuralNet(int[] layerNeurons, String[] layerType, int inputNeurons)
	{
		this.numLayers = layerNeurons.length;
		this.layers = new Layer[numLayers];

		int numInputs;
		//Initializes Fully Connected Neural Network Layers
		for(int i = 0; i < this.numLayers; i++)
		{
			if(i == 0)
			{
				numInputs = inputNeurons; //Number of inputs to the neural network
			}else
			{
				numInputs = layerNeurons[i-1]; //Number of inputs = numper of outputs from previous layer. 
			}
			layers[i] = new Layer(layerType[i], layerNeurons[i], numInputs); // Create layer
		}
	}

	/**
	 * Train the neural network given the parameters.
	 * 
	 * This method uses vector based back propagation to learn.
	 * 
	 * @param dataset 
	 * @param epochs 
	 * @param learningRate 
	 * @param momentum
	 */
	public void train(Dataset dataset, int epochs, double learningRate, double momentum)
	{
		//Epoch iteration
		int count = 0;
		for(int e = 0; e < epochs; e++)
		{
			dataset.randomPerm(); // Randomizes the input dataset after each epoch
			
			/* Batch iteration - currently all of data is used for XOR problem
			 * While this loop is redundant (erroneous), it results in less examples 
			 * needed to achieve 100% accuracy, and thus decreases runtime 10%.
			 * This is most likely an anomaly of the XOR dataset and weights. 
			 */
			for(int i = 0; i < dataset.getNumExamples(); i++)
			{
				Matx data = dataset.getData();
				Matx labels = dataset.getLabels();
				
				//Training Example iteration
				for(int j = 0; j < data.getRows(); j++)
				{
					Matx dataInput = data.getRow(j).getTranspose(); // TODO currently all of data not mini-batch.
					Matx dataInputLabel = labels.getRow(j).getTranspose();
					//Feed Forward
					feedForward(dataInput);
					//Back-propagate error deltas
					backPropagate(dataInputLabel);
					//Update Weights with error deltas
					updateWeights(dataInput,learningRate);
					count++;
				}
			}
			learningRate *= momentum; // Controls the change in the learning rate between epochs
		}
		System.out.println(count);
	}
	
	/**
	 * Feeds a data example through the neural network resulting in a classification output.
	 * Outputs of the hidden layers are stored in the layer objects of the network.
	 * 
	 * @param dataInput Matx example
	 * @return Matx containing the output of the Neural Network given the input
	 */
	private Matx feedForward(Matx dataInput)
	{
		Matx output = dataInput;
		
		for(int k = 0; k < numLayers; k++)
		{
			output = layers[k].computeLayerOutput(output); 
		}
		return output;
	}
	
	/**
	 * Compute the errors in the neural network according to outputs previously 
	 * computed and the label for the data example.
	 * 
	 * @param dataLabel
	 */
	private void backPropagate(Matx dataLabel)
	{
		//Calculate output error delta
		Matx output_delta_error=layers[numLayers-1].computeErrorDelta(dataLabel); //error from output layer

		//Back-propogate errors in hidden layers
		for(int k = numLayers-2; k >= 0; k--)
		{
			output_delta_error = layers[k].computeErrorDelta(output_delta_error, layers[k+1].getWeight());
		}
	}
	
	/**
	 * Update the weights of the neural network according to the input example
	 * and learning rate. 
	 * 
	 * @param data
	 * @param learningRate
	 */
	private void updateWeights(Matx inputExample, double learningRate)
	{
		//Update Weights 
		Matx prevLayerOutput;
		for(int k = numLayers-1; k>=0; k--)
		{
			if(k == 0)
			{
				prevLayerOutput = inputExample;
			}else{
				prevLayerOutput = layers[k-1].getOutput();
			}
			layers[k].updateWeights(learningRate,prevLayerOutput);
		}
	}

	/**
	 * Test a dataset using the current neural network.
	 * 
	 * @param dataset
	 * @param verbose if true: presents more output to the terminal
	 */
	public void test(Dataset dataset, boolean verbose)
	{
		System.out.println("Layer 1 Weights:");
		System.out.println(layers[0].getWeight().toString());
		System.out.println("Layer 2 Weights:");
		System.out.println(layers[1].getWeight().toString());
		
		Matx data = dataset.getData();
		Matx labels = dataset.getLabels();
		
		int classification;
		int label;
		int numCorrect=0;
		int numIncorrect=0;
		//Each Training Example iteration
		for(int j = 0; j < data.getRows(); j++)
		{
			Matx dataInput = data.getRow(j).getTranspose(); // TODO currently all of data not mini-batch.
			Matx dataInputLabel = labels.getRow(j).getTranspose();
			
			Matx output = feedForward(dataInput);
			
			if(output.get(0, 0) > 0.5)
			{
				classification = 1;
			}else{
				classification = 0;
			}
			
			if(dataInputLabel.get(0, 0) > 0.5)
			{
				label = 1;
			}else{
				label = 0;
			}
			
			if(classification == label)
			{
				if(verbose) 
					System.out.println(data.getRow(j).toString() + " " + output + " " + "correct.");
				numCorrect++;
			}else
			{
				if(verbose) 
					System.out.println(data.getRow(j).toString() + " " + output + " " + "incorrect.");
				numIncorrect++;
			}
			
		}
		System.out.println("Accuracy: "+ ((double)numCorrect/(numCorrect+numIncorrect)*100.0) + "%");
	}

}
