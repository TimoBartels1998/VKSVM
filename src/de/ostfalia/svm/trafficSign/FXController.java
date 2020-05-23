package de.ostfalia.svm.trafficSign;

import java.io.IOException;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerMOSSE;
import org.opencv.videoio.VideoCapture;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

//hab diese klasse aus einem tutorial kopiert, weiß nicht wie in wie fern wir das markieren müssen
//bzw wie sehr wir die verändern müssen
/*
 * The controller for our application, where the application logic is
 * implemented. It handles the button for starting/stopping the camera and the
 * acquired video stream.
 *
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @author <a href="http://max-z.de">Maximilian Zuleger</a> (minor fixes)
 * @version 2.0 (2016-09-17)
 * @since 1.0 (2013-10-20)
 *
 */
public class FXController
{
	private static final String POSITIVE_IMAGES = "Images/TrainDataPos";
	private static final String NEGATIVE_IMAGES = "Images/TrainDataNeg";
	
	private static final String POSITIVE_IMAGES_TEST = "Images/TestDataPos";
	private static final String NEGATIVE_IMAGES_TEST = "Images/TestDataNeg";
	private static final String POSITIVE_IMAGE_TEST_CSV_FILE = "Images/TestDataPos/GT-final_test.csv";
	
	private static final String TEST_IMAGE_SRC = "Images/DebugTestImages/TestImg11.jpg";
	private static final String TEST_IMAGE_LOCALISATION_DST = "Images/DebugTestImages/Analyzed_Localisation.bmp";
	private static final String TEST_IMAGE_CLASSIFICATION_DST = "Images/DebugTestImages/Analyzed_Classification.bmp";
	
	private static final String SVM_LOCALISATION_FILE_NAME = "SVM/svmTrained.svm";
	private static final String SVM_CLASSIFICATION_FILE_NAME = "SVM/svmTrainedClassification.svm";
	
	private static final boolean TRAIN_LOCALISATION = false;
	private static final boolean TRAIN_CLASSIFICATION = false;
	
	private static final boolean TEST_LOCALISATION = false;
	private static final boolean TEST_CLASSIFICATION = false;
	
	/**
	 * The frames from the video are resized by the factor 1/VIDEO_SIZE_FACTOR
	 * before the svm threads searches for traffic signs
	 */
	private static final int VIDEO_SIZE_FACTOR = 1;
	
	/**
	 * default value is 33
	 */
	private static final int VIDEO_SPEED = 100;
	
	// the FXML button
	@FXML
	private Button button;
	// the FXML image view
	@FXML
	private ImageView currentFrame;
	
	@FXML
	private ImageView currentAnalyzedFrame;
	
	@FXML
	private ImageView currentFrameColorMask;
	
	@FXML
	private ImageView currentClassifiedFrame;
	
	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that realizes the video capture
	private VideoCapture capture = new VideoCapture("Videos/Video6.mp4");
	// a flag to change the button behavior
	private boolean cameraActive = false;
	
	private SupportVectorMachine spvmLocalisation = null;
	
	private SupportVectorMachine spvmClassification = null;
	
	private PipedInputStream  ControllerFromSVMInput = new PipedInputStream();
	private PipedOutputStream SVMToControllerOutput = new PipedOutputStream();
	
	private PipedInputStream  SVMTFromControllerInput = new PipedInputStream();
	private PipedOutputStream ControllerToSVMOutput = new PipedOutputStream();
	
	private FilterAlgorithm rectangles = new FilterAlgorithm();
	
	private Mat analyzedFrame;
	private Mat frameTmp;
	
	private List<Tracker> trackerList;
	
	private int num = 0;
	
	public void initalizeSVM () {
		SupportVectorMachine.initialize();
		
		//initialize LOCALISATION SVM
		spvmLocalisation = new SupportVectorMachine();
		if(TRAIN_LOCALISATION)
			spvmLocalisation.trainLocalisation(SVM_LOCALISATION_FILE_NAME, "SVM/traindata.bmp", ImageList.loadPositiveOneList(POSITIVE_IMAGES),
				ImageList.loadNegatives(NEGATIVE_IMAGES));
		else
			spvmLocalisation.load(SVM_LOCALISATION_FILE_NAME);
		
		if(TEST_LOCALISATION)
			spvmLocalisation.testPictures(ImageList.loadPositiveTestData(POSITIVE_IMAGES_TEST), ImageList.loadNegatives(NEGATIVE_IMAGES_TEST));
		
		//initialize CLASSIFICATION SVM
		spvmClassification = new SupportVectorMachine();
		if(TRAIN_CLASSIFICATION)
			spvmClassification.trainClassification(SVM_CLASSIFICATION_FILE_NAME, "SVM/traindataClassification.bmp"
					, ImageList.loadPositives(POSITIVE_IMAGES) );
		else
			spvmClassification.load(SVM_CLASSIFICATION_FILE_NAME);
		
		if(TEST_CLASSIFICATION)
			spvmClassification.testPictures(ImageList.loadPositiveTestData(POSITIVE_IMAGES_TEST)
					, ImageList.loadPositiveTestDataClassOrder(POSITIVE_IMAGE_TEST_CSV_FILE));
		
		/*FilterAlgorithm rectangles = new FilterAlgorithm();
		Mat save = spvmLocalisation.compute(Imgcodecs.imread(TEST_IMAGE_SRC), Filter.HEATMAP, rectangles);
		Imgcodecs.imwrite("frameAnalyzed.bmp", save);
		save = Imgcodecs.imread(TEST_IMAGE_SRC);
		for (Rectangle rec : rectangles) {
			Imgproc.rectangle(save, new Point(rec.x, rec.y), new Point(rec.x + rec.radiusX, rec.y + rec.radiusY),
    				new Scalar(0,0,0), 2);
		}
		Imgcodecs.imwrite("frameAnalyzedHeatmap.bmp", save);
		
		save = spvmClassification.classifyTrafficSign(Imgcodecs.imread(TEST_IMAGE_SRC), rectangles);
		Imgcodecs.imwrite("frameAnalyzedClassify.bmp", save);*/
//		Mat testImage = Imgcodecs.imread("Images/DebugTestImages/SignStop.jpg");
//		spvmLocalisation.calcVisualisationOfHOG(testImage);
//		System.exit(1);
	}
	
	/**
	 * The action triggered by pushing the button on the GUI
	 *
	 * @param event
	 *            the push button event
	 */
	@FXML
	protected void startCamera(ActionEvent event)
	{
		if (!this.cameraActive)
		{
//			capture.open("Video1.mp4");
			//connect the pipes for the Thread communication
			try {
				ControllerFromSVMInput = new PipedInputStream();
				SVMToControllerOutput = new PipedOutputStream(ControllerFromSVMInput);
				
				SVMTFromControllerInput = new PipedInputStream(10000000);
				ControllerToSVMOutput = new PipedOutputStream(SVMTFromControllerInput);
				System.out.println("Pipes verbunden");
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
			
			//Create Thread which does the svm calculations on single frames
			new Thread(new Runnable() {
				
				@Override
				public void run() {
					Configuration.debug("SVM Thread started");
					try {
						//signals that svm Thread is ready 
						SVMToControllerOutput.write(1);
						SVMToControllerOutput.flush();
						while(true) {
							//read frame
							while(SVMTFromControllerInput.available()==0) {}
							byte[] bytes = new byte[10000000];
							int bytesRead = 0;
							bytesRead = SVMTFromControllerInput.read(bytes, bytesRead, SVMTFromControllerInput.available());
							Mat frame = Utils.BytesToMat(ByteBuffer.wrap(bytes, 0, bytesRead));
							
							//locate traffic signs
							FilterAlgorithm rectangles = new FilterAlgorithm();
							Mat save = spvmLocalisation.compute(frame, Filter.HEATMAP, rectangles);
							/*for (Rectangle rec : rectangles) {
								Imgproc.rectangle(save, new Point(rec.x, rec.y), new Point(rec.x + rec.radiusX, rec.y + rec.radiusY),
					    				new Scalar(0,0,255));
							}*/
							Imgcodecs.imwrite("frameAnalyzed.bmp", save);
							Image imageToShow = Utils.mat2Image(save);
							updateImageView(currentAnalyzedFrame, imageToShow);
							
							imageToShow = Utils.mat2Image(Imgcodecs.imread("colorMaskInColorRectangle.bmp"));
							updateImageView(currentFrameColorMask, imageToShow);
							
							frame = spvmClassification.classifyTrafficSign(frame, rectangles);
							if(rectangles.size() > 0)
								Imgcodecs.imwrite("Images/DebugAnalyzedImages/frameAnalyzed" + num++ + ".bmp", frame);
							imageToShow = Utils.mat2Image(frame);
							updateImageView(currentClassifiedFrame, imageToShow);
							
							//write rectangles into ouptut pipe
							if(rectangles.size() == 0)
								SVMToControllerOutput.write(1);
							else
								SVMToControllerOutput.write(
										FilterAlgorithm.filterAlgorithmToByteBuffer(rectangles).array());
							
							SVMToControllerOutput.flush();
						}
					} catch (IOException e) {
						e.printStackTrace();
						System.out.println("bla");
					}
				}
			}).start();
			
			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				
				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						updateImageView(currentFrame, imageToShow);
					}
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, VIDEO_SPEED, TimeUnit.MILLISECONDS);
				
				// update the button content
				this.button.setText("Stop Camera");
			}
			else
			{
				// log the error
				System.err.println("Impossible to open the camera connection...");
			}
		}
		else
		{
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.button.setText("Start Camera");
			
			// stop the timer
			this.stopAcquisition();
		}
	}
	
	
	private Mat grabFrame()
	{
		// init everything
		Mat frame = new Mat();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(frame);
				if(ControllerFromSVMInput.available() > 0) {
					byte[] bytes = new byte[ControllerFromSVMInput.available()];
					ControllerFromSVMInput.read(bytes);
					rectangles = new FilterAlgorithm();
					trackerList = new ArrayList<Tracker>();
					if(bytes.length >= Rectangle.BYTES) {
						//read rectangles from
						rectangles = FilterAlgorithm.byteBufferToFilterAlgorithm(ByteBuffer.wrap(bytes)
								, VIDEO_SIZE_FACTOR);
						analyzedFrame = Mat.zeros(frameTmp.size(), frameTmp.type());
						frameTmp.copyTo(analyzedFrame);
						
						for (Rectangle r : rectangles) {
							if(r.radiusX > 1 && r.radiusY > 1) {
								Tracker tmp = TrackerMOSSE.create();
								tmp.init(analyzedFrame, new Rect2d(r.x, r.y, r.radiusX, r.radiusY));
								trackerList.add(tmp);
							}
						}
					}
					
					//send new frame to svm Thread
					frameTmp = Mat.zeros(frame.size(), frame.type());
					frame.copyTo(frameTmp);
					Mat frameSmall = new Mat(frame.rows()/VIDEO_SIZE_FACTOR
							, frame.cols()/VIDEO_SIZE_FACTOR, frame.type());
					Imgproc.resize(frame, frameSmall, frameSmall.size());
					ControllerToSVMOutput.write(Utils.MatToByte(frameSmall).array());
					ControllerToSVMOutput.flush();
				}
				FilterAlgorithm rectanglesTracked = new FilterAlgorithm();
				if(trackerList != null)
					for (Tracker tracker : trackerList) {
						Rect2d rec2d = new Rect2d();
						if(tracker.update(frame, rec2d))
							rectanglesTracked.add(new Rectangle((int)rec2d.x, (int)rec2d.y, (int)rec2d.width, (int)rec2d.height));
					}
				frame = spvmClassification.classifyTrafficSign(frame, rectanglesTracked);
				Imgproc.resize(frame, frame, new Size(frame.cols()/VIDEO_SIZE_FACTOR, frame.rows()/VIDEO_SIZE_FACTOR));
			}
			catch (Exception e)
			{
				// log the error
				System.err.println("Exception during the image elaboration: " );
				e.printStackTrace();
			}
		}
		
		return frame;
	}
	
	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition()
	{
		if (this.timer!=null && !this.timer.isShutdown())
		{
			try
			{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(VIDEO_SPEED, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}
		
		if (this.capture.isOpened())
		{
			// release the camera
			this.capture.release();
		}
	}
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		Utils.onFXThread(view.imageProperty(), image);
	}
	
	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed()
	{
		this.stopAcquisition();
		System.exit(1);
	}
	
}
