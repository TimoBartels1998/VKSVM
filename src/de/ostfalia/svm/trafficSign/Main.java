package de.ostfalia.svm.trafficSign;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;

import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;

public class Main extends Application{
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		launch(args);
	}

	@Override
	public void start(Stage primaryStage) throws Exception {
		ByteBuffer b = Utils.MatToByte(Imgcodecs.imread("Images/DebugTestImages/Sign50.jpg"));
		Imgcodecs.imwrite("test.bmp" ,Utils.BytesToMat(b));
		try
		{
			// load the FXML resource
			FXMLLoader loader = new FXMLLoader(getClass().getResource("FXHelloCV.fxml"));
			// store the root element so that the controllers can use it
			BorderPane rootElement = (BorderPane) loader.load();
			// create and style a scene
			Scene scene = new Scene(rootElement, 800, 600);
			scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
			// create the stage with the given title and the previously created
			// scene
			primaryStage.setTitle("JavaFX meets OpenCV");
			primaryStage.setScene(scene);
			
			FXController controller = loader.getController();
			System.out.println();
			primaryStage.setOnShowing(new EventHandler<WindowEvent>() {
				
				@Override
				public void handle(WindowEvent event) {
					controller.initalizeSVM();
				}
			});
						
			// set the proper behavior on closing the application
			primaryStage.setOnCloseRequest((new EventHandler<WindowEvent>() {
				public void handle(WindowEvent we)
				{
					controller.setClosed();
				}
			}));

			// show the GUI
			primaryStage.show();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		
	}
}
 