package org.easypr.test;

import static org.bytedeco.javacpp.opencv_highgui.imread;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.easypr.core.CoreFunc.getPlateType;

import java.util.Vector;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.easypr.core.CharsRecognise;
import org.easypr.core.CoreFunc.Color;
import org.easypr.core.PlateDetect;
import org.junit.Test;

/**
 * @author lin.yao
 * 
 */
public class EasyPrTest {

    @Test
    public void testPlateDetect() {
        String imgPath = "res/image/test.jpg";

        Mat src = imread(imgPath);
        PlateDetect plateDetect = new PlateDetect();
        plateDetect.setPDLifemode(true);
        Vector<Mat> matVector = new Vector<Mat>();
        if (0 == plateDetect.plateDetect(src, matVector)) {
            for (int i = 0; i < matVector.size(); ++i) {
                try {
                    imshow("Plate Detected", matVector.get(i));
                    System.in.read();
                } catch (Exception ex) {
                }
            }
        }
    }

    @Test
    public void testCharsRecognise() {
        String imgPath = "res/image/test_chars_segment/chars_segment.jpg";

        Mat src = imread(imgPath);
        CharsRecognise cr = new CharsRecognise();
        cr.setCRDebug(true);
        String result = cr.charsRecognise(src);
        System.out.println("Chars Recognised: " + result);
    }

    @Test
    public void testColorDetect() {
        String imgPath = "res/image/test_plate_type/yellow.jpg";

        Mat src = imread(imgPath);

        Color color = getPlateType(src, true);
        System.out.println("Color Deteted: " + color);
    }

}
