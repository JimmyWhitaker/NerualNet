package com.jmw.image.NeuralNet;

public class DropConnectLayer extends Layer
{

	/**
	 * Determines if a de-serialized file is compatible with this class.
	 */
	private static final long serialVersionUID = -1767319077705188292L;
	
	private Matx mask; //TODO change to bit array instead of matrix
	private double maskProbability = 0.5;

	public DropConnectLayer(String activationFunction, int numNeurons, int numInputs)
	{
		super(activationFunction, numNeurons, numInputs);
		//this.mask = Matx.createBinaryRandMatx(weight.getRows(),weight.getCols()); // TODO Add mask probability
	}
	
	/**
	 * Compute the masked output of the layer given an input.
	 * 
	 * @param layerInput
	 * @return masked output of the layer
	 */
	@Override
	public Matx computeLayerOutput(Matx layerInput)
	{
		//mask the weight
		this.mask = Matx.createBinaryRandMatx(weight.getRows(), weight.getCols(),this.maskProbability);
		Matx maskedWeight = Matx.elementMultiply(this.weight, this.mask);
		Matx maskedWeightedInput = Matx.multiply(maskedWeight, layerInput);
		
		//TODO Incorporate bias (does it need its own mask?)
		
		this.weightedInput = maskedWeightedInput;
		this.output = activationFunction.getOutput(maskedWeightedInput);
		return this.output;
	}
	
	/**
	 * Update the weight and bias of the layer.
	 * 
	 * @param learningRate 
	 * @param prevLayerOutput
	 */
	@Override
	public void updateWeights(double learningRate, Matx prevLayerOutput)
	{
		Matx deltaWeight = Matx.multiply(this.error, prevLayerOutput.getTranspose());
		Matx maskedDeltaWeight = Matx.elementMultiply(deltaWeight, this.mask);
		maskedDeltaWeight = Matx.scalarMultiply(learningRate, maskedDeltaWeight);
		this.weight = Matx.add(this.weight, maskedDeltaWeight);
		
		//TODO Update the bias if there is one. 
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
