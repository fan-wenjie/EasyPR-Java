package org.easypr.core;

import org.easypr.util.MatHelper;

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
        } else if (in.channels() == 1) {
            equalizeHist(in, out);
        }
        return out;
    }

    // ！获取垂直和水平方向直方图
    public Mat ProjectedHistogram(Mat img, int t) {
        int sz = (t != 0) ? img.rows() : img.cols();
        Mat mhist = Mat.zeros(1, sz, CV_32F).asMat();

        for (int j = 0; j < sz; j++) {
            Mat data = (t != 0) ? img.row(j) : img.col(j);
            MatHelper.setElement(mhist, countNonZero(data), j);//统计这一行或一列中，非零元素的个数，并保存到mhist中
        }

        //Normalize histogram
        double max = 0;
        for (int j = 0; j < sz; j++)
            if ((Float) MatHelper.getElement(mhist, j) > max)
                max = (Float) MatHelper.getElement(mhist, j);
        if (max > 0)
            mhist.convertTo(mhist, -1, 1.0f / max, 0);//用mhist直方图中的最大值，归一化直方图
        return mhist;
    }


    //! 获得车牌的特征数
    public Mat getTheFeatures(Mat in) {
        final int VERTICAL = 0;
        final int HORIZONTAL = 1;

        //Histogram features
        Mat vhist = ProjectedHistogram(in, VERTICAL);
        Mat hhist = ProjectedHistogram(in, HORIZONTAL);

        //Last 10 is the number of moments components
        int numCols = vhist.cols() + hhist.cols();

        Mat out = Mat.zeros(1, numCols, CV_32F).asMat();

        //Asign values to feature,样本特征为水平、垂直直方图
        int j = 0;
        for (int i = 0; i < vhist.cols(); i++, ++j) {
            byte []buffer = new byte[4];
            vhist.ptr(i).get(buffer);
            out.ptr(j).put(buffer);
        }
        for (int i = 0; i < hhist.cols(); i++, ++j) {
            byte []buffer = new byte[4];
            hhist.ptr(i).get(buffer);
            out.ptr(j).put(buffer);
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
    public Mat getHistogramFeatures(final Mat image) {
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
