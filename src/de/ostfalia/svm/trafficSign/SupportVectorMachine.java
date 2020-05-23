package de.ostfalia.svm.trafficSign;


import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;

/**
 * Support vector machine
 */
public class SupportVectorMachine {
	private static boolean initialized = false;
	
	private HOGDescriptor hog;
	private SVM svm;
	
	/**
	 * Initialize library
	 */
	public static void initialize() {;
		if(!initialized) {
			Configuration.debug("Initialize OpenCV library ...");
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
			initialized = true;
		}
	}
	
	/**
	 * Constructor
	 */
	public SupportVectorMachine() {
		Configuration.debug("Create HOG ...");
		Configuration.debug("  Image size: " + Configuration.IMAGE_SIZE);
		Configuration.debug("  Block size: " + Configuration.BLOCK_SIZE);
		Configuration.debug("  Cell size: " + Configuration.CELL_SIZE);
		hog = new HOGDescriptor(Configuration.IMAGE_SIZE, Configuration.BLOCK_SIZE, Configuration.CELL_SIZE,
				Configuration.CELL_SIZE, 9);
	}
	
	/**
	 * Train SVM for localisation of a traffic sign
	 * @param svmFile File for saving SVM
	 * @param bmpFile File for saving BMP
	 * @param positives List of positive images
	 * @param negatives List of negative images
	 */
	public void trainLocalisation(String svmFile, String bmpFile, ImageList positives, ImageList negatives) {
		long time = System.currentTimeMillis();
		Configuration.debug("Train SVM for localisation...");
		MatOfFloat traindata = new MatOfFloat();
	    Mat trainlabel = new Mat();
	    svm = SVM.create();
	    svm.setKernel(Configuration.KERNEL);
	    svm.setC(Configuration.C_LOCALISATION);
	    svm.setGamma(Configuration.GAMMA_LOCALISATION);
		//Positive images
		Configuration.debug("  Create positive train data ...");
		for (Mat mat : positives) {
			MatOfFloat descriptors = new MatOfFloat();
			/*Mat gray = new Mat(mat.rows(), mat.cols(), CvType.CV_8U, new Scalar(3));
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.resize(gray, dd, new Size(20 , 40));*/
            hog.compute(mat, descriptors);
            Mat descriptorT = new Mat(descriptors.cols(), descriptors.rows(), descriptors.type());
            Core.transpose(descriptors, descriptorT);
            traindata.push_back(descriptorT);
            trainlabel.push_back(Mat.ones(1, 1, CvType.CV_32SC1));
		}
		Configuration.debug("  Positive train data created");
		//Negative images
	    if(negatives == null || negatives.size() == 0) {
	    	svm.setType(SVM.ONE_CLASS);
	    	svm.setNu(Configuration.WO_N_PRECISION);
	    } else {
			Configuration.debug("  Create negative train data ...");
			for (Mat mat : negatives) {
				MatOfFloat descriptors = new MatOfFloat();
				/*Mat gray = new Mat();
	            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);
	            Imgproc.resize(gray, gray, new Size(20 , 40));*/
	            hog.compute(mat, descriptors);
	            Mat descriptorT = new Mat(descriptors.cols(), descriptors.rows(), descriptors.type());
	            Core.transpose(descriptors, descriptorT);
	            traindata.push_back(descriptorT);
	            trainlabel.push_back(Mat.zeros(1, 1, CvType.CV_32SC1));
			}
			Configuration.debug("  Negative train data created");
		}
		//Save trained data
		Configuration.debug("  Size of trained data: " + traindata.size());
		Configuration.debug("  Save trained data to \"" + bmpFile + "\" ...");
	    Imgcodecs.imwrite(bmpFile, traindata);
	    Configuration.debug("  Train SVM and save to \"" + svmFile + "\" ...");
	    svm.train(traindata, Ml.ROW_SAMPLE, trainlabel);
	    svm.save(svmFile);
		Configuration.debug("SVM trained (" + 
				Configuration.deltaTime(time, System.currentTimeMillis()) + ")");
	}
	
	public void trainClassification(String svmFile, String bmpFile, List<ImageList> positives) {
		long time = System.currentTimeMillis();
		Configuration.debug("Train SVM for classification...");
		MatOfFloat traindata = new MatOfFloat();
	    Mat trainlabel = new Mat();
	    svm = SVM.create();
	    svm.setKernel(Configuration.KERNEL);
	    svm.setC(Configuration.C_CLASSIFICATION);
	    svm.setGamma(Configuration.GAMMA_CLASSIFICATION);
	    //svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 10, 1e-6));
		//Positive images
		Configuration.debug("  Create positive train data ...");
		int positivesSize = positives.size();
		for(int i=0; i < positivesSize; i++) {
			Mat labelMat = new Mat(1, 1, CvType.CV_32S);
			labelMat.put(0, 0, i);			
			
			for (Mat mat : positives.get(i)) {
				MatOfFloat descriptors = new MatOfFloat();
				
	            hog.compute(mat, descriptors);
	            Mat descriptorT = new Mat(descriptors.cols(), descriptors.rows(), descriptors.type());
	            Core.transpose(descriptors, descriptorT);
	            traindata.push_back(descriptorT);
	            trainlabel.push_back(labelMat);
			}
		}
		Configuration.debug("  Positive train data created");
	    
		//Save trained data
		Configuration.debug("  Size of trained data: " + traindata.size());
		Configuration.debug("  Save trained data to \"" + bmpFile + "\" ...");
	    Imgcodecs.imwrite(bmpFile, traindata);
	    Configuration.debug("  Train SVM and save to \"" + svmFile + "\" ...");
	    svm.train(traindata, Ml.ROW_SAMPLE, trainlabel);
	    //svm.trainAuto(traindata, Ml.ROW_SAMPLE, trainlabel);
	    svm.save(svmFile);
		Configuration.debug("SVM trained (" + 
				Configuration.deltaTime(time, System.currentTimeMillis()) + ")");
	}
	
	/**
	 * Load SVM
	 * @param svmFile File for loading SVM
	 */
	public void load(String svmFile) {
		long time = System.currentTimeMillis();
		Configuration.debug("Load SVM ...");
		svm = SVM.load(svmFile);
		Configuration.debug("SVM loaded (" + Configuration.deltaTime(time, System.currentTimeMillis()) + ")");
	}

	/**
	 * Compute rectangles with SVM
	 * @param image Image
	 * @return Image with computed rectangles
	 */
	public Mat compute(Mat image, Filter filter, List<Rectangle> found) {
		long time = System.currentTimeMillis();
		Mat newImage = new Mat();
		image.copyTo(newImage);
		FilterAlgorithm filterAlgorithm = new FilterAlgorithm();
		
		int numberOfRects = searchInPictureWithColorFilter(filterAlgorithm, image, newImage, 
				new Scalar(255, 0, 0), 1);
		
		Configuration.debug("Image analysed (numberOfRects: " + numberOfRects + ") (" + Configuration.deltaTime(time, System.currentTimeMillis()) + ")");
		
		List<Rectangle> rectangles;
		switch(filter) {
		case AVERAGE:
			rectangles = filterAlgorithm.filterAverage(); break;
		case OVERLAPPING:
			rectangles = filterAlgorithm.filterOverlapping(true); break;
		case HEATMAP:
			rectangles = filterAlgorithm.filterHeatmap(image, Configuration.OVERLAPPING_RESIZE); break;
		default:
			rectangles = filterAlgorithm;
		}
		found.addAll(rectangles);
		Configuration.debug("  New rectangles calculated (" + rectangles.size() + ")");
		for(Rectangle rectangle : rectangles) {
			Imgproc.rectangle(newImage, rectangle.getRect(), new Scalar(0, 0, 0), 2);
		}
		return newImage;
	}
	
	/**
	 * Load, compute and save image
	 * @see compute(Mat image)
	 * @param loadFile Load image from this file
	 * @param saveFile Save image to this file
	 * @return True if loaded
	 */
	public boolean compute(String loadFile, String saveFile, Filter filter, List<Rectangle> found) {
		Mat load = Imgcodecs.imread(loadFile, Imgcodecs.IMREAD_COLOR);
		if(load.empty()) {
			return false;
		}
		Configuration.debug("Analyse image \"" + loadFile + "\"");
		if(Configuration.DEBUG) {
			System.out.print("  ");
		}
	    Mat save = compute(load, filter, found);
	    Imgcodecs.imwrite(saveFile, save);
	    Configuration.debug("Save analysed image to \"" + saveFile + "\"");
	    return true;
	}
	
	/**
	 * Search for rectangles in picture
	 * @param image Image
	 * @param dest Destination image
	 * @param width Width
	 * @param height Height
	 * @param color Color
	 * @return Rectangles found
	 */
	private int searchInPicture(FilterAlgorithm filter, Mat image, Mat dest, int radius, Scalar color) {
		int found = 0;

		for(int rows = 0; rows + radius <= image.rows(); rows += Configuration.SEARCH_STEPS) {
			for(int cols = 0; cols + radius <= image.cols(); cols += Configuration.SEARCH_STEPS) {
				Mat subMat = image.submat(rows, rows + radius, cols, cols + radius);
				if(predict(subMat) != 0) {
		        	filter.add(new Rectangle(cols, rows, radius, radius));
		        	found++;
		        }
			}
		}
		return found;
	}
	
	private int searchInPictureWithColorFilter(FilterAlgorithm filter, Mat image, Mat dest, Scalar color, double resize) {
		//create colorMask fro traffic Signs
		Mat colorMask = new FilterAlgorithm().filterTrafficSignWithColor(image);
		final Size kernelSize = new Size(20, 20);//20,20
		final Point anchor = new Point(-1, -1);
		final int iterations = 2;
		Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, kernelSize);
		Mat colorMaskInColor = Mat.zeros(image.rows(), image.cols(), image.type());
//		
//		Core.bitwise_and(image, image, colorMaskInColor, colorMask);
//		Imgcodecs.imwrite("colorMask.bmp", colorMask);
//		Imgcodecs.imwrite("colorMaskInColor.bmp", colorMaskInColor);
		
		final Size kernelSizeErode = new Size(4, 4);
		Mat kernelErode = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, kernelSizeErode);
		Imgproc.erode(colorMask, colorMask, kernelErode, anchor);
		Imgproc.dilate(colorMask, colorMask, kernel, anchor, iterations);
//		Core.bitwise_and(image, image, colorMaskInColor, colorMask);
	
//		if(Configuration.DEBUG)
//			Imgcodecs.imwrite("colorMaskDilate.bmp", colorMask);
//		Imgcodecs.imwrite("colorMaskInColorDilate.bmp", colorMaskInColor);
		List<MatOfPoint> contours = new LinkedList<>();
		Imgproc.findContours(colorMask, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_TC89_KCOS);
		
		int found = 0;
		for (MatOfPoint contour : contours) {
			
			Rect rectangle = Imgproc.boundingRect(contour);
			//filter out rectangles, which are too small
			if(rectangle.width < Configuration.SMALL_SQUARE || rectangle.height < Configuration.SMALL_SQUARE)
				continue;
			
			//resize the rectangle by: double resize
			rectangle.x -= Math.max(0, (rectangle.width * (resize - 1) / 2));
			rectangle.y -= Math.max(0, (rectangle.height * (resize - 1) / 2));
			rectangle.width *= resize;
			rectangle.height *= resize;
			if(rectangle.width + rectangle.x > image.cols())
				rectangle.width = image.cols() - rectangle.x;
			if(rectangle.height + rectangle.y > image.rows())
				rectangle.height = image.rows() - rectangle.y;
			
			Imgproc.rectangle(colorMask, rectangle,
    				color, -1);
			Imgproc.rectangle(colorMaskInColor, rectangle,
    				new Scalar(255, 255, 255), -1);
//			Core.bitwise_and(image, image, colorMaskInColor, colorMask);
//			Imgcodecs.imwrite("colorMaskRectangle.bmp", colorMask);
//			Imgcodecs.imwrite("colorMaskInColorRectangle.bmp", colorMaskInColor);
			Mat newImage = new Mat();
			Mat subMat = image.submat(rectangle);
			subMat.copyTo(newImage);
			
			FilterAlgorithm filterInner = new FilterAlgorithm();
			
			found += searchInPicture(filterInner, subMat, newImage, Configuration.LARGE_SQUARE, 
					new Scalar(255, 0, 0));
			found += searchInPicture(filterInner, subMat, newImage, Configuration.MIDDLE_SQUARE, 
					new Scalar(0, 255, 0));
			found += searchInPicture(filterInner, subMat, newImage, Configuration.SMALL_SQUARE, 
					new Scalar(0, 0, 255));
			
			for (Rectangle rectangle2 : filterInner) {
				rectangle2.x += rectangle.x;
				rectangle2.y += rectangle.y;
			}
			filter.addAll(filterInner);
			
		}
		Core.bitwise_and(image, colorMaskInColor, colorMaskInColor);
//		Core.bitwise_and(image, image, colorMaskInColor, colorMask);
//		Imgcodecs.imwrite("colorMaskRectangle.bmp", colorMask);
		Imgcodecs.imwrite("colorMaskInColorRectangle.bmp", colorMaskInColor);
		if(Configuration.DEBUG) {
			for (Rectangle rec : filter) {
				Imgproc.rectangle(dest, new Point(rec.x, rec.y), new Point(rec.x + rec.radiusX, rec.y + rec.radiusY),
	    				color);
			}
    	}
		Imgcodecs.imwrite("frameAnalyzedLocal.bmp", dest);
		
		return found;
	}
	
	/**
	 * 
	 * @param positives
	 * @param negatives
	 * @return
	 */
	public double testPictures(ImageList positives, ImageList negatives) {
		int rightPos = 0;
		for (Mat mat : positives) {
			if(predict(mat) != 0) {
	        	rightPos++;
	        }
		}
		if(positives.size() != 0)
			Configuration.debug(rightPos + " von " + positives.size() + "erkannt (" + (double)(rightPos)/(double)(positives.size()) + ")");
		
		int rightNeg = 0;
		if(negatives != null && negatives.size() != 0) {
			for (Mat mat : negatives) {
				if(predict(mat) == 0) {
		        	rightNeg++;
		        }
			}
			Configuration.debug(rightNeg + " von " + negatives.size() + "erkannt (" + (double)(rightNeg)/(double)(negatives.size()) + ")");
		}
		
		return ( (double)(rightNeg)/(double)(negatives.size()) + (double)(rightPos)/(double)(positives.size()) ) / 2.0;
	}
	
	public double testPictures(ImageList positives, List<Integer> response) {
		int rightPos = 0;
		Iterator<Integer> iter = response.iterator();
		for (Mat mat : positives) {
			if(iter.hasNext() && predict(mat) == iter.next()) {
	        	rightPos++;
	        }
		}
		if(positives.size() != 0)
			Configuration.debug(rightPos + " von " + positives.size() + "erkannt (" + (double)(rightPos)/(double)(positives.size()) + ")");
		
		return (double)(rightPos)/(double)(positives.size());
	}
	
	/**
	 * predicts a class for an image with the SVM
	 * @param image
	 * @return
	 */
	public double predict(Mat image) {
		MatOfFloat descriptors = new MatOfFloat();
	    Imgproc.resize(image, image, Configuration.IMAGE_SIZE);
	    hog.compute(image, descriptors);
	    Mat descriptorT = new Mat(descriptors.cols(), descriptors.rows(), descriptors.type());
        Core.transpose(descriptors, descriptorT);
        return svm.predict(descriptorT);
	}
	
	/**
	 * classifies the Traffic Signs in the rectangles
	 * saves an image with an rectangle around every sign and the name of that traffic sign under
	 * the rectangle.
	 * @param src
	 * @param rectangles
	 * @return
	 */
	public void classifyTrafficSign(String srcDirectory, String saveDirectory, List<Rectangle> rectangles) {
		Mat mat = Imgcodecs.imread(srcDirectory, Imgcodecs.IMREAD_COLOR);
		mat = classifyTrafficSign(mat, rectangles);
		Imgcodecs.imwrite(saveDirectory, mat);
	}
	
	/**
	 * classifies the Traffic Signs in the rectangles
	 * returns an image with an rectangle around every sign and the name of that traffic sign under
	 * the rectangle.
	 * @param src
	 * @param rectangles
	 * @return
	 */
	public Mat classifyTrafficSign(Mat src, List<Rectangle> rectangles) {
		Mat image = new Mat();
		src.copyTo(image);
		
		for (Rectangle rectangle : rectangles) {
			Mat subImage = image.submat(Math.max(rectangle.y, 0), Math.min(rectangle.y + rectangle.radiusY, image.rows())
					, Math.max(rectangle.x, 0), Math.min(rectangle.x + rectangle.radiusX, image.cols()));
			
			int signNumber = (int)predict(subImage);
			Trafficsign sign = Trafficsign.getByValue(signNumber);
			
			String name1 = sign.toString(), name2 = "";
			if(name1.length() > 15) {
				name2 = "-" + name1.substring(name1.length()/2);
				name1 = name1.substring(0, name1.length()/2);
				
				Imgproc.putText(image, name2
						, new Point(rectangle.x, rectangle.y + rectangle.radiusY + 30)
						, Imgproc.FONT_HERSHEY_SIMPLEX, 0.25, new Scalar(0,0,255));
			}
			Imgproc.putText(image, name1
					, new Point(rectangle.x, rectangle.y + rectangle.radiusY + 15)
					, Imgproc.FONT_HERSHEY_SIMPLEX, 0.25, new Scalar(0,0,255));
			
			Imgproc.rectangle(image, new Point(rectangle.x, rectangle.y)
					, new Point(rectangle.x + rectangle.radiusX, rectangle.y + rectangle.radiusY)
					, new Scalar(0,0,0), 4);
		}
		
		return image;
	}
	
	public void calcVisualisationOfHOG(Mat image) {
		Imgproc.resize(image, image, Configuration.IMAGE_SIZE);
		
		final int numberOfCellsX = (int)(Configuration.IMAGE_SIZE.width / Configuration.CELL_SIZE.width);
		final int numberOfCellsY = (int)(Configuration.IMAGE_SIZE.height / Configuration.CELL_SIZE.height);
		
		Mat gradients = new Mat();
	    Mat angles = new Mat();
	    hog.computeGradient(image, gradients, angles);
	    List<Mat> matSplit = new ArrayList<Mat>();
	    Core.split(gradients, matSplit);

	    Imgcodecs.imwrite("gradients1.bmp", matSplit.get(0));
	    Imgcodecs.imwrite("gradients2.bmp", matSplit.get(1));
	    Core.bitwise_or(matSplit.get(0), matSplit.get(1), gradients);
	    Imgcodecs.imwrite("gradients.bmp", gradients);
	    Imgcodecs.imwrite("subImage.bmp", image);
	    //Imgcodecs.imwrite("test.bmp", angles);
		
	    
	    
		MatOfFloat descriptors = new MatOfFloat();
	    Imgproc.resize(image, image, Configuration.IMAGE_SIZE);
	    hog.compute(image, descriptors);
	    
	    double[][][] his = new double[numberOfCellsX][numberOfCellsY][9];
	    
	    final int cellsInBlockX = 2;
	    final int cellsInBlockY = 2;
	    final int numberOfBlocksY = numberOfCellsY - cellsInBlockY + 1;
	    final int numberOfBlocksX = numberOfCellsX - cellsInBlockX + 1;
	    int counter = 0;
	   
    	for (int x = 0; x < numberOfBlocksX; x++) {
    		 for (int y = 0; y < numberOfBlocksY; y++) {
    			 for (int blockX = 0; blockX < cellsInBlockX; blockX++) {
    				 for (int blockY = 0; blockY < cellsInBlockY; blockY++) {
	    			
//	    				System.out.println("y: " + y + ", x: " + x + ", BlockY: " 
//	    	    				+ blockY + ", BlockX: " + blockX + ", counter: " + counter + " - " + (counter+8));
	    				for (int bin = 0; bin < 9; bin++) {
	    					his[x+blockX][y+blockY][bin] = descriptors.get(counter++, 0)[0];
						}
					}
				}
	    	}
	    }
//	    System.out.println("counter: " + counter);
	    
	    final int cellHeight = 30;
	    final int cellWidth = 30;
	    final int lineRadius = 15;
	    Mat hogVis = Mat.zeros(cellHeight * numberOfCellsY
				, cellWidth * numberOfCellsX, image.type());
//	    Mat hogVis = image;
//	    Imgproc.resize(hogVis, hogVis,new Size(cellWidth * numberOfCellsX, cellHeight * numberOfCellsY));
	    
	    for (int i = 0; i < numberOfCellsY; i++) {
	    	for (int j = 0; j < numberOfCellsX; j++) {
				for (int j2 = 0; j2 < 9; j2++) {
					int angleDegree = j2*20;
					if(angleDegree > 90)
						angleDegree = 180 - angleDegree;
					double[] dirVector = new double[2];
					dirVector[0] = Math.sin(Math.toRadians(angleDegree));
					dirVector[1] = Math.cos(Math.toRadians(angleDegree));
					
					if(j2*20 < 90) {
						dirVector[0] *= -1;
					} else {
						dirVector[1] *= -1;
					}
					
					Point offset = new Point(j * cellWidth + cellWidth/2, i * cellHeight + cellHeight/2);
					
//					Mat tmp = Mat.zeros(hogVis.rows(), hogVis.cols(), hogVis.type());
					Point a = new Point(offset.x + dirVector[0] * lineRadius, offset.y + dirVector[1] * lineRadius);
					Point b = new Point(offset.x - dirVector[0] * lineRadius, offset.y - dirVector[1] * lineRadius);
					Imgproc.line(hogVis, a, b, new Scalar(255*his[j][i][j2], 255*his[j][i][j2], 255*his[j][i][j2]), 1);
//					Core.bitwise_not(tmp, tmp);
//					Core.bitwise_and(hogVis, tmp, hogVis);
					if(his[j][i][j2] > 0.01) {
						System.out.println("winkel: " + angleDegree);
					}
				}
			}
		}
	    Imgcodecs.imwrite("hogVis.png", hogVis);
	}
}
