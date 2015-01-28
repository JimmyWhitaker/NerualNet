package com.jmw.image.NeuralNet;

/**
 * NeuralNet Implementation
 * App.java
 * Purpose: Main class to run a Neural Network Tests.
 * 
 * @author Jimmy Whitaker
 * @version 0.1 1/21/15
 */
public class App 
{
	/**
	 * The main method for the NeuralNet program.
	 *
	 * @param args Not used
	 */
	public static void main( String[] args )
	{
		//Compute Runtime
		long start = System.currentTimeMillis();

		//Run test
		mnistTest();

		System.out.println("Runtime: " + (System.currentTimeMillis()-start));
	}

	/**
	 * Mnist Test
	 */
	public static void mnistTest()
	{
		int[] neuronsPerLayer = {300,10};
		String[] layerType = {"Sigmoid","Softmax"}; // Currently all neurons in a layer have the same activation function
		int epochs = 100;
		int batchSize = 128;
		double learningRate = 0.1;
		double momentum = 1;
		String filename = "Mnist-300-DC-Sig.ser";
		
		// Import training data
		MnistDataset mnistTrainingDataset = MnistDataset.load("trainingData.ser");
		
		//Import testing data
		MnistDataset mnistTestingDataset = MnistDataset.load("testingData.ser");

//		NeuralNet nn = NeuralNet.load(filename);
		DropConnectNeuralNet dc = new DropConnectNeuralNet(neuronsPerLayer, layerType, mnistTrainingDataset.getNumImageFeatures());
		
		//Train the classifier
		dc.train(mnistTrainingDataset, mnistTestingDataset, epochs, batchSize, learningRate, momentum, filename);
	}
	
	/**
	 * Start an MNIST Test from scratch.
	 */
	public static void fullMnistTest()
	{
		int[] neuronsPerLayer = {300,10};
		String[] layerType = {"Sigmoid","Softmax"}; // Currently all neurons in a layer have the same activation function
		int epochs = 100;
		int batchSize = 128;
		double learningRate = 0.1;
		double momentum = 1;
		String filename = "Mnist-300-100-10-Sig.ser";
		
		//Import Training Data
		MnistDataset mnistTrainingDataset = new MnistDataset("train-images-idx3-ubyte","train-labels-idx1-ubyte");

		//Import Testing Data
		MnistDataset mnistTestingDataset = new MnistDataset("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte");
		
		//Create new Neural Network
		NeuralNet nn = new NeuralNet(neuronsPerLayer, layerType, mnistTrainingDataset.getNumImageFeatures());
		
		//Train Neural Network
		nn.train(mnistTrainingDataset, mnistTestingDataset, epochs, batchSize, learningRate, momentum, filename);
	}

	/**
	 * XOR Test
	 */
	public static void xorTest()
	{
		int[] neuronsPerLayer = {2,1};
		String[] layerType = {"Tanh","Tanh"}; // Currently all neurons in a layer have the same activation function
		int epochs = 10000;
		int batchSize = 2;
		double learningRate = 0.7;
		double momentum = 1.0;
		String filename = "xor.ser";

		double[][] xor_data = {{0,0,1,1},{0,1,0,1}};
		double[] xor_labels = {0,1,1,0};

		MatxDataset dataset = new MatxDataset(Matx.createMatx(xor_data).getTranspose(), Matx.createMatx(xor_labels));
		//TODO add mean and stddev to parameters
		NeuralNet nn = new NeuralNet(neuronsPerLayer, layerType, dataset.getData().getCols()); 
		
		nn.train(dataset, dataset, epochs, batchSize, learningRate, momentum, filename);
	}
	
	/**
	 * Manual Back-propagation Test
	 */
	public static void backpropTest()
	{
		//Error Discovery Test - Comparison by Hand
//		int[] neuronsPerLayer = {2,1};
//		String[] layerType = {"Sigmoid","Sigmoid"};
//		double[] data = {1,0,1};
//		double[] label = {1};
//		MatxDataset dataset = new MatxDataset(Matx.createMatx(data).getTranspose(), Matx.createMatx(label));
//		NeuralNet nn = new NeuralNet(neuronsPerLayer, layerType, dataset.getData().getCols());
//		double[][] w1 = {{0.2,0.4,-0.5},{-0.3,0.1,0.2}};
//		double[] w2 = {-0.3,-0.2};
//		nn.layers[0].weight = Matx.createMatx(w1);
//		nn.layers[1].weight = Matx.createMatx(w2).getTranspose();
//		double[][] b1 = {{-0.4},{0.2}};
//		double[][] b2 = {{0.1}};
//		nn.layers[0].bias = Matx.createMatx(b1);
//		nn.layers[1].bias = Matx.createMatx(b2);
//		nn.train(dataset,dataset, 1, 1, 0.9, 1,null);
//		nn.test(dataset,false);
	}
}
