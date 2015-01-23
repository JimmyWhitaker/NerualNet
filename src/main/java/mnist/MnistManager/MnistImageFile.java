package mnist.MnistManager;

import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * 
 * MNIST database image file. Contains additional header information for the
 * number of rows and columns per each entry.
 * 
 * Source: https://code.google.com/p/mnist-tools/
 */
public class MnistImageFile extends MnistDbFile {
    public int rows;
    public int cols;

    /**
     * Creates new MNIST database image file ready for reading.
     * 
     * @param name
     *            the system-dependent filename
     * @param mode
     *            the access mode
     * @throws IOException
     * @throws FileNotFoundException
     */
    public MnistImageFile(String name, String mode) throws FileNotFoundException, IOException {
        super(name, mode);

        // read header information
        rows = readInt();
        cols = readInt();
    }

    /**
     * Reads the image at the current position.
     * 
     * @return matrix representing the image
     * @throws IOException
     */
    public int[][] readImage() throws IOException {
        int[][] dat = new int[getRows()][getCols()];
        for (int i = 0; i < getCols(); i++) {
            for (int j = 0; j < getRows(); j++) {
                dat[i][j] = readUnsignedByte();
            }
        }
        return dat;
    }
    
    /**
     * Reads the image at the current position and processes it.
     * Processing: converts values to double and re-scales (divides by 255).
     * 
     * @return matrix representing the image
     * @throws IOException
     */
    public double[] readProcessedImage() throws IOException {
    	int datLength = getRows()*getCols();
    	int value = 0;
        double[] dat = new double[datLength];
        for (int i = 0; i < datLength; i++) {
            value = readUnsignedByte();
            dat[i] = (double)value/255.0;
        }
        return dat;
    }

    /**
     * Move the cursor to the next image.
     * 
     * @throws IOException
     */
    public void nextImage() throws IOException {
        super.next();
    }

    /**
     * Move the cursor to the previous image.
     * 
     * @throws IOException
     */
    public void prevImage() throws IOException {
        super.prev();
    }

    @Override
    protected int getMagicNumber() {
        return 2051;
    }

    /**
     * Number of rows per image.
     * 
     * @return int
     */
    public int getRows() {
        return rows;
    }

    /**
     * Number of columns per image.
     * 
     * @return int
     */
    public int getCols() {
        return cols;
    }

    @Override
    public int getEntryLength() {
        return cols * rows;
    }

    @Override
    public int getHeaderSize() {
        return super.getHeaderSize() + 8; // to more integers - rows and columns
    }
}