package com.jmw.image.NeuralNet;

public class DropConnectNeuralNet extends NeuralNet
{
	
	/**
	 * Determines if a de-serialized file is compatible with this class.
	 */
	private static final long serialVersionUID = 4569131899036164711L;

	public DropConnectNeuralNet(int[] layerNeurons, String[] layerType, int inputNeurons, int batchSize) {
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
			if( layerType[i].equals("Softmax") )
			{
				layers[i] = new Layer(layerType[i], layerNeurons[i], numInputs, batchSize); // Create Softmax layer
			}else{
				layers[i] = new DropConnectLayer(layerType[i], layerNeurons[i], numInputs, batchSize);
			}
		}
	}
	
	
	/**
	 * Test a dataset using the DropConnect neural network.
	 * 
	 * @param dataset
	 * @param verbose if true: presents more output to the terminal
	 */
	@Override
	public double test(Dataset testingSet, int batchSize, boolean verbose)
	{
		double threshold = 0.5;
		int classification;
		int label;
		int numCorrect=0;
		int numIncorrect=0;

		testingSet.splitIntoBatches(batchSize);
		Dataset batch = null;
		
		//Testing Batch iteration
		for(int b = 0; b < testingSet.getNumBatches(); b++)
		{
			batch = testingSet.getBatch(); // Get a batch of the data		
			Matx output = testingFeedForward(batch.getData().getTranspose()).getTranspose();
			Matx labels = batch.getLabels();

			//Each Training Example iteration
			for(int i = 0; i < labels.getRows(); i++)
			{
				//If last layer is Softmax
				if( layers[layers.length-1].getActivationFunctionType().equals("Softmax") )
				{
					/**
					 * Get the maximum of an output vector.
					 * Find the location of the 1 in the label vector.
					 * If the output vector contains the max at the 
					 * same index, then it is a correct classification.
					 */
					double max = output.maxInRow(i);
					for(int j = 0; j < labels.getCols(); j++)
					{
						if(labels.get(i, j) == 1.0) 
						{
							if(output.get(i, j) == max && max > 1.0/(double)labels.getCols()) // Not all the same value
							{
								numCorrect++;
								break;
							}else{
								numIncorrect++;
								break;
							}
						}
					}
				}
				else //Output layer isn't softmax
				{
					if(output.get(i, 0) > threshold)
					{
						classification = 1;
					}else{
						classification = 0;
					}

					if(labels.get(i, 0) > threshold)
					{
						label = 1;
					}else{
						label = 0;
					}


					if(classification == label)
					{
						if(verbose) 
							System.out.println(batch.getData().getRow(i) + " " + String.format("%.02f",output.get(i, 0)) + " " + "correct.");
						numCorrect++;
					}else
					{
						if(verbose) 
							System.out.println(batch.getData().getRow(i) + " " + String.format("%.02f",output.get(i, 0)) + " " + "incorrect.");
						numIncorrect++;
					}
				}

			}
		}
		double accuracy = ((double)numCorrect/(numCorrect+numIncorrect)*100.0);
		System.out.println("Accuracy: "+ String.format("%.02f%%",accuracy));
		return accuracy;
	}

	/**
	 * Feed forward for test without masking any weights.
	 * @param dataInput input data
	 * @return output Matx from the Feed Forward Test
	 */
	private Matx testingFeedForward(Matx dataInput)
	{
		Matx output = dataInput;

		for(int k = 0; k < numLayers; k++)
		{
			if(layers[k].activationFunction.getType().equals("Softmax"))
			{
				output = layers[k].computeLayerOutput(output);
			}else{
				output = ((DropConnectLayer)layers[k]).computeUnmaskedLayerOutput(output); 
			}
		}
		return output;
	}

}
