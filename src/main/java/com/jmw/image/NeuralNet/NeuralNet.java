package com.jmw.image.NeuralNet;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/**
 * Fully Connected Neural Network
 * 
 * @author Jimmy Whitaker
 */
public class NeuralNet implements Serializable
{ 
	/**
	 * Determines if a de-serialized file is compatible with this class.
	 */
	private static final long serialVersionUID = 2304156243561498598L;
	
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
	 * This method uses vector based back propagation to learn one example
	 * at a time. 
	 * 
	 * @param dataset 
	 * @param epochs 
	 * @param learningRate 
	 * @param momentum
	 */
	
	public void train(Dataset dataset, Dataset testset, int epochs, int batchSize, double learningRate, double momentum)
	{
		//Import testing data
		Matx testingData = testset.getData();
		Matx testingLabels = testset.getLabels();
		MatxDataset testDataset = new MatxDataset(testingData, testingLabels);
		
		dataset.splitIntoBatches(batchSize);
		
		//Epoch iteration
		for(int e = 0; e < epochs; e++)
		{
			Dataset batch = dataset.getBatch(); // Get a batch of the data
			
			batch.randomPerm(); // Randomizes the batch. TODO randomize entire file rather than just the batch
			
			//Training Batch iteration
			for(int j = 0; j < dataset.getNumBatches(); j++)
			{
				Matx dataInput = batch.getData().getTranspose();
				Matx dataInputLabel = batch.getLabels().getTranspose();
				//Feed Forward
				feedForward(dataInput);
				//Back-propagate error deltas
				backPropagate(dataInputLabel);
				//Update Weights with error deltas
				updateWeights(dataInput,learningRate);
//				showTrainingPercentage(count, dataset.getNumBatches(),epochs);
			}
			//Save Neural Network when testing accuracy is above a certain threshold (avoids over-training)
			if(test(testingData,testingLabels,false)>90)
			{
				save("Mnist-90.ser");
			}
				
			learningRate *= momentum; // Controls the change in the learning rate between epochs
		}
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
	public double test(Matx data, Matx labels, boolean verbose) //TODO change back to Dataset paramenter instead of 2 Matxs
	{
//		Matx data = dataset.getData();
//		Matx labels = dataset.getLabels();
		Matx output = feedForward(data.getTranspose()).getTranspose();
		
		double threshold = 0.5;
		int classification;
		int label;
		int numCorrect=0;
		int numIncorrect=0;
		
		//Each Training Example iteration
		for(int i = 0; i < labels.getRows(); i++)
		{
			//If last layer is Softmax
			if( layers[layers.length-1].getType().equals("Softmax") )
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
						if(output.get(i, j) == max)
						{
							numCorrect++;
						}else{
							numIncorrect++;
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
						System.out.println(data.getRow(i) + " " + String.format("%.02f",output.get(i, 0)) + " " + "correct.");
					numCorrect++;
				}else
				{
					if(verbose) 
						System.out.println(data.getRow(i) + " " + String.format("%.02f",output.get(i, 0)) + " " + "incorrect.");
					numIncorrect++;
				}
			}

		}
		double accuracy = ((double)numCorrect/(numCorrect+numIncorrect)*100.0);
		System.out.println("Accuracy: "+ String.format("%.02f%%",accuracy));
		return accuracy;
	}
	
	/**
	 * Show the completed training percentage.
	 * @param numCompleted
	 * @param numBatches
	 * @param epochs
	 */
	private static final void showTrainingPercentage(int numCompleted, int numBatches, int epochs)
	{
		double percentage = ( ((double)numCompleted) / (numBatches*epochs) *100);
		System.out.println(String.format("%.02f%%",percentage) +'\r' );
	}
	
	/**
	 * Save a serialized version of the NeuralNet
	 * 
	 * @param filename name (location) of ouput file typically ending with (.ser)
	 */
	public void save(String filename)
	{
		try
		{
			FileOutputStream fileOut = new FileOutputStream(filename);
			ObjectOutputStream outStream = new ObjectOutputStream(fileOut);
			outStream.writeObject(this);
			outStream.close();
			fileOut.close();
		}catch(IOException i)
		{
			i.printStackTrace();
		}
	}
	/**
	 * De-serialize NeuralNet object.
	 * @param filename location of file
	 * @return de-serialized NeuralNet 
	 */
	public static NeuralNet load(String filename)
	{
		//Deserialize
		System.out.println("************");
		System.out.println("Deserializing");
		NeuralNet nn = null;
		try
		{
			FileInputStream fileIn =new FileInputStream(filename);
			ObjectInputStream in = new ObjectInputStream(fileIn);
			nn = (NeuralNet) in.readObject();
			in.close();
			fileIn.close();
		}catch(IOException i)
		{
			i.printStackTrace();
			return null;
		}catch(ClassNotFoundException c)
		{
			System.out.println("NeuralNet class not found.");
			c.printStackTrace();
			return null;
		}
		System.out.println("Done Deserializing.");
		System.out.println("Number of Layers: " + nn.layers.length);
		return nn;   
	}

}
