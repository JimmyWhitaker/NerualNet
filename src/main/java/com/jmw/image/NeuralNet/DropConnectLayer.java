package com.jmw.image.NeuralNet;

public class DropConnectLayer extends Layer
{

	/**
	 * Determines if a de-serialized file is compatible with this class.
	 */
	private static final long serialVersionUID = -1767319077705188292L;
	
	private Matx mask; //TODO change to bit array instead of matrix
	private double maskProbability = 0.5;

	public DropConnectLayer(String activationFunction, int numNeurons, int numInputs, int batchSize)
	{
		super(activationFunction, numNeurons, numInputs,batchSize);
		//this.mask = Matx.createBinaryRandMatx(weight.getRows(),weight.getCols()); // TODO Add mask probability
	}
	
	/**
	 * Compute the masked output of the layer given an input.
	 * 
	 * o = a((M.*W)*v)
	 * 
	 * where,
	 * o is output
	 * a() is activation function
	 * M is the mask matrix
	 * W is the weigh matrix
	 * v is the data vector (or matrix) with a 1s bias input
	 * and .* is element-wise multiplication
	 * 
	 * @param layerInput
	 * @return masked output of the layer
	 */
	@Override
	public Matx computeLayerOutput(Matx layerInput)
	{
		//mask the weight
		this.mask = Matx.createBinaryRandMatx(weight.getRows(), weight.getCols(), this.maskProbability);
		Matx maskedWeight = Matx.elementMultiply(this.weight, this.mask);
		Matx maskedWeightedInput = Matx.multiply(maskedWeight, layerInput.appendRow(this.bias));
		
		this.weightedInput = maskedWeightedInput;
		this.output = activationFunction.getOutput(maskedWeightedInput);
		return this.output;
	}
	
	/**
	 * Update the weight of the layer.
	 * 
	 * Ws = Ws - n(M.*A'w)
	 * where, 
	 * Ws is the weight matrix of the layer
	 * n is the learning rate
	 * M is the mask matrix
	 * A'w is the gradient of the loss for the next layer
	 * 
	 * @param learningRate 
	 * @param prevLayerOutput
	 */
	@Override
	public void updateWeights(double learningRate, Matx prevLayerOutput)
	{
		Matx deltaWeight = Matx.multiply(this.error, prevLayerOutput.appendRow(this.bias).getTranspose());
		Matx maskedDeltaWeight = Matx.elementMultiply(deltaWeight, this.mask);
		maskedDeltaWeight = Matx.scalarMultiply(learningRate, maskedDeltaWeight);
		this.weight = Matx.add(this.weight, maskedDeltaWeight);
	}
	
	/**
	 * Compute the testing (unmasked) output of the layer given an input.
	 * 
	 * @param layerInput
	 * @return output of the layer
	 */
	public Matx computeUnmaskedLayerOutput(Matx layerInput)
	{
		return super.computeLayerOutput(layerInput);
	}
}
