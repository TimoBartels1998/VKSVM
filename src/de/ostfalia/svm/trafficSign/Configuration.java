package de.ostfalia.svm.trafficSign;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.ml.SVM;

/**
 * Configuration
 */
public class Configuration {
	/**
	 * Output debug messages
	 */
	public static final boolean DEBUG = true;
	
	/**
	 * Kernel of SVM
	 */
	public static final int KERNEL = SVM.RBF;
	
	/**
	 * Image size
	 */
	public static final Size IMAGE_SIZE = new Size(48, 80);
	/**
	 * Block size
	 */
	public static final Size BLOCK_SIZE = new Size(16, 16);
	
	/**
	 * Cell size
	 */
	public static final Size CELL_SIZE = new Size(8, 8);
	
	/**
	 * Small square for computing rectangles in picture
	 */
	public static final int VERY_SMALL_SQUARE = 20;
	
	/**
	 * Small square for computing rectangles in picture
	 */
	public static final int SMALL_SQUARE = 30;//40
	
	/**
	 * Middle square for computing rectangles in picture
	 */
	public static final int MIDDLE_SQUARE = 40;//60
	
	/**
	 * Large square for computing rectangles in picture
	 */
	public static final int LARGE_SQUARE = 50;//80
	
	/**
	 * Color of rectangles in picture
	 */
	public static final Scalar FOUND_RECT_COLOR = new Scalar(255, 0, 0);
	
	/**
	 * Precision for training without negatives
	 * Values: 0-1 (0 = accurate, 1 = inaccurate)
	 */
	public static final double WO_N_PRECISION = 0.9;
	
	/**
	 * Threshold of heat map
	 * @see Filter.drawHeatmap(Mat image, double resize)
	 */
	public static final int OVERLAPPING_THRESHOLD = 2;
	
	public static final int SEARCH_STEPS = 5;
	
	/**
	 * Resize for rectangles in heat map
	 * @see Filter.drawHeatmap(Mat image, double resize)
	 */
	public static final double OVERLAPPING_RESIZE = 1.0;
	
	public static final double GAMMA_LOCALISATION = 5.0625000000000009e-01;
	
	public static final double C_LOCALISATION = 2.5;
	
	public static final double GAMMA_CLASSIFICATION = 2.3750000000000002e-02;
	
	public static final double C_CLASSIFICATION = 6.2500000000000000e+01;
	
	/**
	 * Output debug message
	 * @param text Debug message
	 */
	public static void debug(String text) {
		if(DEBUG) {
			System.out.println(text);
		}
	}
	
	/**
	 * Get delta time string
	 * @param time1 Time 1
	 * @param time2 Time 2
	 * @return Delta time string
	 */
	public static String deltaTime(long time1, long time2) {
		long millis = time2 - time1;
		long seconds = millis / 1000;
		millis = millis % 1000;
		long minutes = 0;
		if(seconds > 60) {
			minutes = seconds / 60;
			seconds = seconds % 60;
		}
		StringBuilder builder = new StringBuilder();
		addTimeString(builder, minutes, "min");
		addTimeString(builder, seconds, "s");
		addTimeString(builder, millis, "ms");
		return builder.toString();
	}
	
	/**
	 * Add time string to delta time
	 * @param builder String builder
	 * @param value Time value
	 * @param unit Time unit
	 */
	private static void addTimeString(StringBuilder builder, long value, String unit) {
		if(value > 0) {
			if((builder.length() != 0) && (builder.charAt(builder.length() - 1) != ' ')) {
				builder.append(' ');
			}
			builder.append(value);
			builder.append(unit);
		}
	}
}
