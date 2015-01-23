package com.jmw.image.NeuralNet;

/**
 * Relu (Rectified Linear Unit) Activation Function
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
		double E_x = Math.exp(1d*net);                
		return Math.log10(1d + E_x);               
	}

	@Override
	protected double getDerivative(double net) {
		double den = 1d + Math.exp(-1.0*net);

		return (1d / den);
	}

	@Override
	public String getType()
	{
		return "Relu";
	}	

}

