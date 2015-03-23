package org.easypr.core;

import org.easypr.util.Convert;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/*
 * Created by fanwenjie
 * @version 1.1
 */


public class Features implements SVMCallback{

    public Mat histeq(Mat in) {
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

    // ！获取垂直和水平方向直方图
    public float[][] ProjectedHistogram(Mat img) {

        float []nonZeroHor = new float[img.rows()];
        float []nonZeroVer = new float[img.cols()];
        for(int i=0;i<img.rows();++i)
            for(int j=0;j<img.cols();++j)
                if(0!=img.ptr(i,j).get()){
                    ++nonZeroHor[i];
                    ++nonZeroVer[j];
                }
        float [][]out = new float[][]{nonZeroVer,nonZeroHor};
        for(int i=0;i<2;++i){
            float max = 0;
            for(int j=0;j<out[i].length;++j)
                if(max<out[i][j])
                    max = out[i][j];
            if(max>0)
                for(int j=0;j<out[i].length;++j)
                    out[i][j]/=max;
        }
        return out;
    }


    //! 获得车牌的特征数
    public Mat getTheFeatures(Mat in) {

        //Histogram features
        float [][]hist = ProjectedHistogram(in);
        float[] vhist = hist[0];
        float[] hhist = hist[1];

        //Last 10 is the number of moments components
        int numCols = vhist.length + hhist.length;

        Mat out = Mat.zeros(1, numCols, CV_32FC1).asMat();

        //Asign values to feature,样本特征为水平、垂直直方图
        int j = 0;
        for (int i = 0; i < vhist.length; i++, ++j) {
            out.ptr(j).put(Convert.getBytes(vhist[i]));
        }
        for (int i = 0; i < hhist.length; i++, ++j) {
            out.ptr(j).put(Convert.getBytes(hhist[i]));
        }
        return out;
    }

    // ! EasyPR的getFeatures回调函数
// ！本函数是生成直方图均衡特征的回调函数
    @Override
    public Mat getHisteqFeatures(final Mat image) {
        return histeq(image);
    }

    // ! EasyPR的getFeatures回调函数
// ！本函数是获取垂直和水平的直方图图值
    @Override
    public Mat getHistogramFeatures(Mat image) {
        Mat grayImage = new Mat();
        cvtColor(image, grayImage, CV_RGB2GRAY);
        //grayImage = histeq(grayImage);
        Mat img_threshold = new Mat();
        threshold(grayImage, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
        return getTheFeatures(img_threshold);
    }


    // ! EasyPR的getFeatures回调函数
// ！本函数是获取SITF特征子
// !
    @Override
    public Mat getSIFTFeatures(final Mat image) {
        //TODO: 待完善
        return null;
    }


    // ! EasyPR的getFeatures回调函数
// ！本函数是获取HOG特征子
    @Override
    public Mat getHOGFeatures(final Mat image) {
        //TODO: 待完善
        return null;
    }
}
