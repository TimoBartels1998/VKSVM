package de.ostfalia.svm.trafficSign;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * Heat map
 */
@SuppressWarnings("serial")
public class FilterAlgorithm extends LinkedList<Rectangle> {
	
	/**
	 * minimal color for blue traffic signs
	 */
//	private static final Scalar MIN_BULE_COLOR_TRAFFIC_SIGN = new Scalar(150, 70, 0);
//	private static final Scalar MIN_BULE_COLOR_TRAFFIC_SIGN = new Scalar(96, 40, 47);
	private static final Scalar MIN_BULE_COLOR_TRAFFIC_SIGN = new Scalar(99, 84, 22);
	
	/**
	 * maximal color for blue traffic signs
	 */
//	private static final Scalar MAX_BULE_COLOR_TRAFFIC_SIGN = new Scalar(255, 180, 110);
//	private static final Scalar MAX_BULE_COLOR_TRAFFIC_SIGN = new Scalar(125, 255, 255);
	private static final Scalar MAX_BULE_COLOR_TRAFFIC_SIGN = new Scalar(121, 255, 255);
	
	/**
	 * minimal color for red traffic signs
	 */
//	private static final Scalar MIN_RED_COLOR_TRAFFIC_SIGN = new Scalar(0, 0, 100);
//	private static final Scalar MIN_RED_COLOR_TRAFFIC_SIGN_1 = new Scalar(168, 40, 47);
	private static final Scalar MIN_RED_COLOR_TRAFFIC_SIGN_1 = new Scalar(166, 91, 51);
	
	/**
	 * maximal color for red traffic signs
	 */
//	private static final Scalar MAX_RED_COLOR_TRAFFIC_SIGN = new Scalar(70, 70, 255);
//	private static final Scalar MAX_RED_COLOR_TRAFFIC_SIGN_1 = new Scalar(179, 255, 255);
	private static final Scalar MAX_RED_COLOR_TRAFFIC_SIGN_1 = new Scalar(179, 255, 252);
	
	/**
	 * minimal color for red traffic signs
	 */
//	private static final Scalar MIN_RED_COLOR_TRAFFIC_SIGN = new Scalar(0, 0, 100);
//	private static final Scalar MIN_RED_COLOR_TRAFFIC_SIGN_2 = new Scalar(0, 40, 47);
	private static final Scalar MIN_RED_COLOR_TRAFFIC_SIGN_2 = new Scalar(0, 91, 51);
	
	/**
	 * maximal color for red traffic signs
	 */
//	private static final Scalar MAX_RED_COLOR_TRAFFIC_SIGN = new Scalar(70, 70, 255);
//	private static final Scalar MAX_RED_COLOR_TRAFFIC_SIGN_2 = new Scalar(9, 255, 255);
	private static final Scalar MAX_RED_COLOR_TRAFFIC_SIGN_2 = new Scalar(6, 255, 252);
	
	
	/**
	 * Summarize all overlapping rectangles and draw summarized rectangle
	 * @param corner True for calculating corners, False for calculating edges
	 */
	public List<Rectangle> filterOverlapping(boolean corner) {
		List<Rectangle> found = new LinkedList<>();
		//Find all overlapping rectangles for one rectangle
		LinkedList<LinkedList<Rectangle>> overlappingLists = new LinkedList<>();
		for(Rectangle rectangle : this) {
			LinkedList<Rectangle> list = new LinkedList<>();
			list.add(rectangle);
			for(Rectangle test : this) {
				if(!rectangle.equals(test) && rectangle.overlap(test)) {
					list.add(test);
				}
			}
			if(list.size() > 1) {
				overlappingLists.add(list);
			}
		}
		//Join all overlapping lists
		for(int i = 0; i < overlappingLists.size(); i++) {
			LinkedList<Rectangle> current = overlappingLists.get(i);
			for(int j = i + 1; j < overlappingLists.size();) {
				if(hasEqualElements(current, overlappingLists.get(j))) {
					join(current, overlappingLists.get(j));
					overlappingLists.remove(j);
				} else {
					j++;
				}
			}
		}
		//Draw rectangles
		for(LinkedList<Rectangle> list : overlappingLists) {
			int minX = Integer.MAX_VALUE;
			int minY = Integer.MAX_VALUE;
			int maxX = 0;
			int maxY = 0;
			for(Rectangle rectangle : list) {
				if(corner) {
					if(minX > rectangle.x && minY > rectangle.y) {
						minX = rectangle.x;
						minY = rectangle.y;
					}
					if(maxX < (rectangle.x + rectangle.radiusX) && maxY < (rectangle.y + rectangle.radiusY)) {
						maxX = rectangle.x + rectangle.radiusX;
						maxY = rectangle.y + rectangle.radiusY;
					}
				} else {
					if(minX > rectangle.x) {
						minX = rectangle.x;
					}
					if(minY > rectangle.y) {
						minY = rectangle.y;
					}
					if(maxX < (rectangle.x + rectangle.radiusX)) {
						maxX = rectangle.x + rectangle.radiusX;
					}
					if(maxY < (rectangle.y + rectangle.radiusY)) {
						maxY = rectangle.y + rectangle.radiusY;
					}
				}
			}
			found.add(new Rectangle(minX, minY, maxX - minX, maxY - minY));
		}
		return found;
	}
	
	/**
	 * Join lists of overlapping rectangles
	 * @param dest Destination list
	 * @param src Source list
	 */
	private void join(LinkedList<Rectangle> dest, LinkedList<Rectangle> src) {
		for(Rectangle rectangle : src) {
			boolean isInDest = false;
			for(int i = 0; i < dest.size(); i++) {
				if(rectangle.equals(dest.get(i))) {
					isInDest = true;
					break;
				}
			}
			if(!isInDest) {
				dest.add(rectangle);
			}
		}
	}
	
	/**
	 * Have two lists of rectangles any equal element?
	 * @param list1 List 1
	 * @param list2 List 2
	 * @return True if has any equal element, False else
	 */
	private boolean hasEqualElements(LinkedList<Rectangle> list1, LinkedList<Rectangle> list2) {
		for(Rectangle rectangle : list1) {
			for(Rectangle test : list2) {
				if(rectangle.equals(test)) {
					return true;
				}
			}
		}
		return false;
	}
	
	/**
	 * Calculate and draw average rectangle
	 */
	public List<Rectangle> filterAverage() {
		List<Rectangle> found = new LinkedList<>();
		//draw average rectangle
		int count = 0;
		double minX = 0;
		double minY = 0;
		double maxX = 0;
		double maxY = 0;
		for(Rectangle rectangle : this) {
			minX += rectangle.x;
			minY += rectangle.y;
			maxX += (rectangle.x + rectangle.radiusX);
			maxY += (rectangle.y + rectangle.radiusY);
			count++;
		}
		minX /= count;
		minY /= count;
		maxX /= count;
		maxY /= count;
		found.add(new Rectangle((int) minX, (int) minY, (int) (maxX - minX), (int) (maxY - minY))); 
		return found;
	}
	
	/**
	 * Calculate heat points and draw rectangles around them
	 * @param image Image
	 * @param resize Resize value of rectangles
	 */
	public List<Rectangle> filterHeatmap(Mat image, double resize) {
		List<Rectangle> found = new LinkedList<Rectangle>();
		//Find heat points
		Mat heatmap = Mat.zeros(image.rows(), image.cols(), image.type());
		for(Rectangle rectangle : this) {
			Mat rectMap = Mat.zeros(image.rows(), image.cols(), image.type());
			Imgproc.rectangle(rectMap, new Point(rectangle.x, rectangle.y),
					new Point(rectangle.x + rectangle.radiusX, rectangle.y + rectangle.radiusY),
					new Scalar(0, 0, 1), -1);
			Core.add(heatmap, rectMap, heatmap);
		}
//		Imgcodecs.imwrite("heatmapRedBefore.bmp", heatmap);
		//Filter heat points with threshold
		Mat thresholds = Mat.zeros(image.rows(), image.cols(), image.type());
		Imgproc.rectangle(thresholds, new Point(0, 0), new Point(thresholds.cols(), thresholds.rows()), 
				new Scalar(0, 0, Configuration.OVERLAPPING_THRESHOLD), -1);
		Core.subtract(heatmap, thresholds, heatmap);
//		Imgcodecs.imwrite("heatmapRedAfter.bmp", heatmap);
		Core.multiply(heatmap, new Scalar(255, 255, 255), heatmap);
		//Draw a rectangle around heat points
		Imgproc.cvtColor(heatmap, heatmap, Imgproc.COLOR_BGR2GRAY);
		List<MatOfPoint> contours = new LinkedList<>();
		Imgproc.findContours(heatmap, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_TC89_KCOS);
		for(MatOfPoint contour : contours) {
			Rect rectangle = Imgproc.boundingRect(contour);
			rectangle.x -= (rectangle.width * (resize - 1) / 2);
			rectangle.y -= (rectangle.height * (resize - 1) / 2);
			rectangle.width *= resize;
			rectangle.height *= resize;
			found.add(new Rectangle(rectangle.x, rectangle.y, rectangle.width, rectangle.height));
		}
		return found;
	}
	
	/**
	 * filters the picture for colors which are in the range from colorMin to colorMax
	 * @param image
	 * @param colorMin
	 * @param colorMax
	 * @return
	 */
	public Mat filterColor(Mat image, Scalar colorMin, Scalar colorMax){
		Mat mask = new Mat(image.rows(), image.cols(), CvType.CV_8U, new Scalar(3));
		Core.inRange(image, colorMin, colorMax, mask);
		return mask;
	}
	
	/**
	 * Uses the filterColor function to show only the pixels in the Mat which have the
	 * color of a traffic sign.
	 * @return
	 */
	public Mat filterTrafficSignWithColor(Mat image) {
		Mat imageHSV = new Mat();
		Imgproc.cvtColor(image, imageHSV, Imgproc.COLOR_BGR2HSV);
		Mat blueMask = filterColor(imageHSV, MIN_BULE_COLOR_TRAFFIC_SIGN, MAX_BULE_COLOR_TRAFFIC_SIGN);
		
		Mat blue = new Mat(image.size(), image.type());
		Mat blueBlue = new Mat(image.size(), image.type());
		blue.setTo(new Scalar(255,0,0));
		Mat blueRedMask = new Mat(image.size(), image.type());
		Core.bitwise_and(blue, blue, blueBlue, blueMask);
		
		Mat redMask = filterColor(imageHSV, MIN_RED_COLOR_TRAFFIC_SIGN_1, MAX_RED_COLOR_TRAFFIC_SIGN_1);
		
		Mat red = new Mat(image.size(), image.type());
		Mat redRed = new Mat(image.size(), image.type());
		red.setTo(new Scalar(0,0,255));
		Core.bitwise_and(red, red, redRed, redMask);
		
		Core.bitwise_or(redRed, blueBlue, blueRedMask);
		Imgcodecs.imwrite("MaskColor.bmp", blueRedMask);
		
		Core.bitwise_or(redMask, blueMask, blueMask);
		redMask = filterColor(imageHSV, MIN_RED_COLOR_TRAFFIC_SIGN_2, MAX_RED_COLOR_TRAFFIC_SIGN_2);
		Core.bitwise_or(redMask, blueMask, blueMask);
		return blueMask;
	}
	
	public static ByteBuffer filterAlgorithmToByteBuffer(FilterAlgorithm list) {
		ByteBuffer bb = ByteBuffer.allocate(list.size() * Rectangle.BYTES); 
		for (Rectangle rectangle : list) {
			bb.putInt(rectangle.x);
			bb.putInt(rectangle.y);
			bb.putInt(rectangle.radiusX);
			bb.putInt(rectangle.radiusY);
		}
		return bb;
	}
	
	public static FilterAlgorithm byteBufferToFilterAlgorithm(ByteBuffer bb, int resize) {
		FilterAlgorithm filter = new FilterAlgorithm();
		int numOfRecs = bb.limit()/Rectangle.BYTES;
		for (int i = 0; i < numOfRecs; i++) {
			int recIndex = i*Rectangle.BYTES;
			filter.add(new Rectangle(bb.getInt(i*16+0)*resize
					, bb.getInt(recIndex+Integer.BYTES)*resize
					, bb.getInt(recIndex+Integer.BYTES*2)*resize
					, bb.getInt(recIndex+Integer.BYTES*3)*resize));
		}
		return filter;
	}
}
