package mnist.MnistManager;

import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * 
 * MNIST database label file.
 * 
 * Source: https://code.google.com/p/mnist-tools/
 */
public class MnistLabelFile extends MnistDbFile {

    /**
     * Creates new MNIST database label file ready for reading.
     * 
     * @param name
     *            the system-dependent filename
     * @param mode
     *            the access mode
     * @throws IOException
     * @throws FileNotFoundException
     */
    public MnistLabelFile(String name, String mode) throws FileNotFoundException, IOException {
        super(name, mode);
    }

    /**
     * Reads the integer at the current position.
     * 
     * @return integer representing the label
     * @throws IOException
     */
    public int readLabel() throws IOException {
        return readUnsignedByte();
    }
    
    /**
     * Provides an array containing a 1 in the index position of the read value.
     * Ex.
     *  Read 7
     *  index: 0 1 2 3 4 5 6 7 8 9
     *  value:[0 0 0 0 0 0 0 1 0 0]
     *  
     * @return integer representing the label
     * @throws IOException
     */
    public double[] readProcessedLabel() throws IOException {
        int index = readUnsignedByte();
        double[] result = new double[10]; //initially all 0s
        result[index] = 1;
        return result;
    }

    @Override
    protected int getMagicNumber() {
        return 2049;
    }
}
