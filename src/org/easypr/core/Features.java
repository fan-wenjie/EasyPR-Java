package org.easypr.core;

import static org.bytedeco.javacpp.opencv_core.merge;
import static org.bytedeco.javacpp.opencv_core.split;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.easypr.core.CoreFunc.features;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;

/**
 * 
 * @author Created by fanwenjie
 * @author lin.yao
 *
 */
public class Features implements SVMCallback {

    /*
     * (non-Javadoc)
     * 
     * @see org.easypr.core.SVMCallback#getHisteqFeatures(org.bytedeco.javacpp.
     * opencv_core.Mat)
     */
    @Override
    public Mat getHisteqFeatures(final Mat image) {
        return histeq(image);
    }

    /*
     * (non-Javadoc)
     * 
     * @see
     * org.easypr.core.SVMCallback#getHistogramFeatures(org.bytedeco.javacpp
     * .opencv_core.Mat)
     */
    @Override
    public Mat getHistogramFeatures(Mat image) {
        Mat grayImage = new Mat();
        cvtColor(image, grayImage, CV_RGB2GRAY);

        Mat img_threshold = new Mat();
        threshold(grayImage, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);

        return features(img_threshold, 0);
    }

    /*
     * (non-Javadoc)
     * 
     * @see
     * org.easypr.core.SVMCallback#getSIFTFeatures(org.bytedeco.javacpp.opencv_core
     * .Mat)
     */
    @Override
    public Mat getSIFTFeatures(final Mat image) {
        // TODO: 待完善
        return null;
    }

    /*
     * (non-Javadoc)
     * 
     * @see
     * org.easypr.core.SVMCallback#getHOGFeatures(org.bytedeco.javacpp.opencv_core
     * .Mat)
     */
    @Override
    public Mat getHOGFeatures(final Mat image) {
        // TODO: 待完善
        return null;
    }

    private Mat histeq(Mat in) {
        Mat out = new Mat(in.size(), in.type());
        if (in.channels() == 3) {
            Mat hsv = new Mat();
            MatVector hsvSplit = new MatVector();
            cvtColor(in, hsv, CV_BGR2HSV);
            split(hsv, hsvSplit);
            equalizeHist(hsvSplit.get(2), hsvSplit.get(2));
            merge(hsvSplit, hsv);
            cvtColor(hsv, out, CV_HSV2BGR);
            hsv = null;
            hsvSplit = null;
            System.gc();
        } else if (in.channels() == 1) {
            equalizeHist(in, out);
        }
        return out;
    }
}
