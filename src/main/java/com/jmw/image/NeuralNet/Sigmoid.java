package com.jmw.image.NeuralNet;

/**
 * Sigmoid Activation Function
 * 
 * @author Jimmy
 */
public class Sigmoid extends ActivationFunction
{
	/**
	 * Constructs new Sigmoid Activation Function
	 */
	public Sigmoid()
	{
		
	}

	@Override
	protected double getOutput(double net) {
		// conditional logic helps to avoid overflow and underflow
		if (net > 100) {
			return 1.0;
		}else if (net < -100) {
			return 0.0;
		}

		double den = 1d + Math.exp(-1.0*net);
		return (1d / den);
	}

	@Override
	protected double getDerivative(double net) {
		// +0.1 is fix for flat spot see http://www.heatonresearch.com/wiki/Flat_Spot
		double output = getOutput(net);
		double derivative = output * (1d - output);
		
		return derivative;
	}

}

