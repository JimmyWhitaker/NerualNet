package com.jmw.image.NeuralNet;

/**
 * Relu (Rectified Linear Unit) Activation Function
 * 
 * @author Jimmy
 */
public class Relu extends ActivationFunction
{
	/**
	 * Constructs new Relu Activation Function
	 */
	public Relu()
	{
		
	}
	
	@Override
	protected double getOutput(double net) {
		// conditional logic helps to avoid overflow and underflow
		if (net > 100) {
			return 1.0;
		}else if (net < -100) {
			return -1.0;
		}

		double E_x = Math.exp(1d*net);                
		return Math.log10(1d + E_x);               
	}

	@Override
	protected double getDerivative(double net) {
		// conditional logic helps to avoid overflow and underflow
		if (net > 100) {
			return 1.0;
		}else if (net < -100) {
			return 0.0;
		}

		double den = 1d + Math.exp(-1.0*net);

		return (1d / den);
	}	

}

