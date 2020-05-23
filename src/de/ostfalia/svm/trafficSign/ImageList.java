package de.ostfalia.svm.trafficSign;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * Image list
 */
@SuppressWarnings("serial")
public class ImageList extends LinkedList<Mat> {
	/**
	 * Empty image list
	 */
	public static final ImageList EMPTY = null;
	
	/**
	 * Get positive image list
	 * @param directory Directory
	 * @return Positive image list
	 */
	public static ImageList loadPositiveOneList(String directory) {
		List<ImageList> list = loadPositives(directory);
		ImageList sumOfAllLists = new ImageList();
		
		for (ImageList imgList : list) {
			sumOfAllLists.addAll(imgList);
		}
		
		return sumOfAllLists;
	}
	
	/**
	 * Gets the list of lists for one traffic sign
	 * @param directory
	 * @return
	 */
	public static List<ImageList> loadPositives(String directory) {
		long time = System.currentTimeMillis();
		Configuration.debug("Load positive images ...");
		LinkedList<ImageList> list = new LinkedList<ImageList>();
		int folder = 0;
		int files = 0;
		ImageList returnList = ImageList.EMPTY;
		
		do {
			String trafficSignDir = directory + "/" + String.format("%05d", folder);
			returnList = loadImagesForOneTrafficSign(trafficSignDir);
			if(returnList.size()!=0)
				list.add(returnList);
			
			files += returnList.size();
			folder++;
		} 
		while(returnList.size() != 0);
		
		Configuration.debug("Positive images loaded (" + Configuration.deltaTime(time, System.currentTimeMillis())
		+ ", " + (folder-1) + "folders , " + files + " files)");
		
		return list;
	}
	
	/**
	 * loads all positives Images for one traffic sign
	 * @param directory
	 * @return
	 */
	public static ImageList loadImagesForOneTrafficSign(String directory) {
		long time = System.currentTimeMillis();
		Configuration.debug("Load positives images for traffic sign " + directory.substring(directory.lastIndexOf('/')) + " ...");
		ImageList list = new ImageList();
		Mat image = Mat.ones(1, 1, CvType.CV_8U);
		int files = 0;
		
		while(!image.empty()) {
			String filePrefix = String.format("%05d", files/30);
			String filePostfix = String.format("%05d", files%30);
			String fileName = filePrefix + "_" + filePostfix + ".ppm";
			image = Imgcodecs.imread(directory + "/" + fileName);
			
			if(!image.empty()) {
				Imgproc.resize(image, image, Configuration.IMAGE_SIZE);
				list.add(image);
			}
			files++;
		}
		Configuration.debug("Positive images for traffic sign " + directory.substring(directory.lastIndexOf('/')) + " loaded (" + Configuration.deltaTime(time, System.currentTimeMillis())
		+ ", " + list.size() + " files)");
		
		return list;
	}
	
	/**
	 * Get negative image list
	 * @param directory Directory
	 * @return Negative image list
	 */
	public static ImageList loadNegatives(String directory) {
		long time = System.currentTimeMillis();
		Configuration.debug("Load negative images ...");
		ImageList list = new ImageList();
		Mat image = Mat.ones(1, 1, CvType.CV_8U);
		int file = 1;
		while(!image.empty()) {
			image = Imgcodecs.imread(directory + "/" + file + ".jpg", Imgcodecs.IMREAD_COLOR);
			file++;
			if(!image.empty()) {
				for(int x = 0; x < 5; x++) {
					for (int y = 0; y < 5; y++) {
						Mat sub = image.submat(image.rows() / 5 * x, image.rows() / 5 * (x+1), 
								image.cols() / 5 * y, image.cols() / 5 * (y+1));
						Imgproc.resize(sub, sub, Configuration.IMAGE_SIZE);
						list.add(sub);
					}
				}
				for(int x = 0; x < 10 && x < image.rows(); x++) {
					for (int y = 0; y < 10 && x < image.cols(); y++) {
						Mat sub = image.submat(image.rows() / 10 * x, image.rows() / 10 * (x+1), 
								image.cols() / 10 * y, image.cols() / 10 * (y+1));
						Imgproc.resize(sub, sub, Configuration.IMAGE_SIZE);
						list.add(sub);
					}
				}
				for(int x = 0; x < 20 && x < image.rows(); x++) {
					for (int y = 0; y < 20 && x < image.cols(); y++) {
						Mat sub = image.submat(image.rows() / 20 * x, image.rows() / 20 * (x+1),
								image.cols() / 20 * y, image.cols() / 20 * (y+1));
						Imgproc.resize(sub, sub, Configuration.IMAGE_SIZE);
						list.add(sub);
					}
				}
			}
		}
		Configuration.debug("Negative images loaded (" + Configuration.deltaTime(time, System.currentTimeMillis())
				+ ", " + (file - 2) + " files)");
		return list;
	}
	
	/**
	 * loads the positive test images
	 * @param directory
	 * @return
	 */
	public static ImageList loadPositiveTestData(String directory) {
		long time = System.currentTimeMillis();
		Configuration.debug("Load positive images ...");
		ImageList list = new ImageList();
		Mat image = Mat.ones(1, 1, CvType.CV_8U);
		
		int file = 0;
		while(!image.empty()) {
			image = Imgcodecs.imread(directory + "/" + String.format("%05d", file) + ".ppm", Imgcodecs.IMREAD_COLOR);
			if(!image.empty()) {
				Imgproc.resize(image, image, Configuration.IMAGE_SIZE);
				list.add(image);
			}
			file++;
		}
		
		Configuration.debug("Positive images loaded (" + Configuration.deltaTime(time, System.currentTimeMillis())
		+ ", " + (file - 1) + " files)");
		return list;
	}
	
	public static List<Integer> loadPositiveTestDataClassOrder(String csvFile) {
        List<Integer> classOfData = new LinkedList<Integer>();
        String line = "";
        String cvsSplitBy = ";";
        
        try(BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
        	//Kopfzeile lesen
        	br.readLine();
        	
        	//Datenzeilen lesen
            while ((line = br.readLine()) != null) {
            	String[] lineSplit = line.split(cvsSplitBy);
            	classOfData.add(Integer.valueOf(lineSplit[7]));
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (NumberFormatException e) {
            e.printStackTrace();
        }
        
        return classOfData;
	}
}
