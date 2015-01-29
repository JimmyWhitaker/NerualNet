package com.jmw.image.NeuralNet;

/**
 * Noisy Relu (Rectified Linear Unit) Activation Function
 * 
 * f(x) = max(0,x + N(0,1))
 * f'(x) = 1 if x > 0
 *         0 otherwise
 * 
 * @author Jimmy
 */
public class Relu extends ActivationFunction
{
	/**
	 * Determines if a de-serialized file is compatible with this class.
	 */
	private static final long serialVersionUID = 5370409328503675226L;

	/**
	 * Constructs new Relu Activation Function
	 */
	public Relu()
	{
		
	}
	
	@Override
	protected double getOutput(double net) {
		return Math.max(0.0, net+Math.random());
	}

	@Override
	protected double getDerivative(double net) {
		if (net > 0) 
			return 1.0;
		else
			return 0.0;
		
	}

	@Override
	public String getType()
	{
		return "Relu";
	}	

}

