package com.jmw.image.NeuralNet;

/**
 * Activation Function used for a layer in a neural network.
 * 
 * @author Jimmy Whitaker
 */
public abstract class ActivationFunction
{
	/**
	 * Computes an element-wise output of the activation function.
	 * 
	 * @param matx input Matx
	 * @return computed output Matx
	 */
	// TODO make this more efficient
	final public Matx getOutput(Matx matx) 
	{
		Matx result = new Matx(matx); 
		double net = 0;
		for(int i = 0; i < matx.getRows(); i++)
		{
			for(int j = 0; j < matx.getCols(); j++)
			{
				net = matx.get(i, j);
				result.set(i,j,getOutput(net));
			}
		}
		return result;                
	}
	
	/**
	 * Computes the output of the activation function for a 
	 * single input.
	 * 
	 * @param net input value
	 * @return ouput value
	 */
	protected abstract double getOutput(double net);
	
	/**
	 * Computes an element-wise derivative output of the activation function.
	 * 
	 * @param matx input Matx
	 * @return computed output Matx
	 */
	//TODO make this more efficient
	public Matx getDerivative(Matx matx)
	{
		Matx result = new Matx(matx); 
		double net = 0;
		for(int i = 0; i < matx.getRows(); i++)
		{
			for(int j = 0; j < matx.getCols(); j++)
			{
				net = matx.get(i, j);
				result.set(i,j,getDerivative(net));
			}
		}
		return result; 
	}
	
	/**
	 * Computes the value of the derivative of the activation function for a 
	 * single input.
	 * 
	 * @param net input value
	 * @return ouput value
	 */
	protected abstract double getDerivative(double net);
}
