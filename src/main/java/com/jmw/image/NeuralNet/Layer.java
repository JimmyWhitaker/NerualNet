package com.jmw.image.NeuralNet;

import java.io.Serializable;

/**
 * Neural Network Layer
 * 
 * @author Jimmy Whitaker
 *
 */
public class Layer implements Serializable
{
	/**
	 * Determines if a de-serialized file is compatible with this class.
	 */
	protected static final long serialVersionUID = 5215917274187351523L;
	
	protected Matx weightedInput; // weight*layerInput
	protected Matx output;
	protected Matx weight; //Bias is incorporated as additional in weights
	protected Matx error; // Error delta
	public Matx bias; // Bias vector of 1s appended to input data
	
	// TODO create mask matrix for drop connect
	protected ActivationFunction activationFunction;
	
	/**
	 * Constructs a new Layer for a Neural Network.
	 * 
	 * @param activationFunction String representation of an activation function.
	 * @param numNeurons Number of neurons in the layer
	 * @param numInputs Number of inputs to the layer
	 */
	public Layer(String activationFunction, int numNeurons, int numInputs, int batchSize)
	{		
		if(activationFunction.equals("Sigmoid"))
		{
			this.activationFunction = new Sigmoid();
		} else if(activationFunction.equals("Tanh"))
		{
			this.activationFunction = new Tanh();
		}else if(activationFunction.equals("Relu"))
		{
			this.activationFunction = new Relu();
		}else if(activationFunction.equals("Softmax"))
		{
			this.activationFunction = new Softmax();
		}
		
		//Randomly initialize the weight Matx according to a normal distribution
		this.weight = Matx.createRandNormMatx(numNeurons, numInputs+1); // Bias incorporated into weight
		this.bias = Matx.createOnesMatx(1, batchSize); //Appended to input matrix
	}
	
	/**
	 * Compute the output of the layer given an input.
	 * 
	 * @param layerInput
	 * @return output of the layer
	 */
	public Matx computeLayerOutput(Matx layerInput)
	{
		this.weightedInput = Matx.multiply(this.weight, layerInput.appendRow(this.bias));
		this.output = activationFunction.getOutput(this.weightedInput);
		return this.output;
	}

	/**
	 * Compute error delta for output layer.
	 * Store value in layer and return it.
	 * 
	 * @param dataLabel
	 * @return error
	 */
	public Matx computeErrorDelta(Matx dataLabel)
	{	
		// Cross Entropy Loss for Softmax output layer
		if(activationFunction.getType().equals("Softmax") )
		{
			this.error = Matx.subtract(dataLabel,this.output); // o - y
		}else{ // Single class output
			Matx term1 = Matx.subtract(dataLabel,this.output);
			Matx term2 = activationFunction.getDerivative(this.weightedInput);
			this.error =  Matx.elementMultiply(term2, term1);
		}
		return this.error;
	}

	/**
	 * Compute error delta for hidden layer.
	 * Store value in layer and return it.
	 * 
	 * @param nextLayerError
	 * @param nextLayerWeight
	 * @return error
	 */
	public Matx computeErrorDelta(Matx nextLayerError, Matx nextLayerWeight)
	{
		//Remove bias from the nextLayerWeight and calculate 
		Matx term1 = Matx.multiply(nextLayerWeight.getTranspose().removeLastRow(), nextLayerError);
		Matx term2 = activationFunction.getDerivative(this.weightedInput);
		this.error = Matx.elementMultiply(term1, term2);
		return this.error;
	}
	
	/**
	 * Update the weight and bias of the layer.
	 * 
	 * @param learningRate 
	 * @param prevLayerOutput
	 */
	public void updateWeights(double learningRate, Matx prevLayerOutput)
	{
		Matx deltaWeight = Matx.multiply(this.error, prevLayerOutput.appendRow(this.bias).getTranspose());
		deltaWeight = Matx.scalarMultiply(learningRate, deltaWeight);
		this.weight = Matx.add(this.weight, deltaWeight);
	}
	
	/**
	 * @return current weight of the layer
	 */
	public Matx getWeight()
	{
		return this.weight;
	}

	/**
	 * @return current output of the layer
	 */
	public Matx getOutput()
	{
		return this.output;
	}
	
	public int getNumNeurons()
	{
		return this.weight.getRows();
	}
	
	public String getActivationFunctionType()
	{
		return this.activationFunction.getType();
	}
	
}
