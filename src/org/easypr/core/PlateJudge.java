package org.easypr.core;

import java.util.Vector;

import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_ml.*;
import static org.bytedeco.javacpp.opencv_core.*;

/*
 * Created by fanwenjie
 * @version 1.1
 */

public class PlateJudge {

    public PlateJudge() {
        loadModel();
    }

    public void loadModel() {
        loadModel(path);
    }

    public void loadModel(String s) {
        svm.clear();
        svm.load(s, "svm");
    }

    //! 直方图均衡
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


    //! 对单幅图像进行SVM判断
    public int plateJudge(final Mat inMat) {
        Mat features = this.features.getHistogramFeatures(inMat);
        //通过直方图均衡化后的彩色图进行预测
        Mat p = features.reshape(1, 1);
        p.convertTo(p, CV_32FC1);
        return (int) svm.predict(p);
    }


    //! 对多幅图像进行SVM判断
    public int plateJudge(Vector<Mat> inVec, Vector<Mat> resultVec) {
        for (int j = 0; j < inVec.size(); j++) {
            Mat inMat = inVec.get(j);
            if (1 == plateJudge(inMat))
                resultVec.add(inMat);
        }
        return 0;
    }

    public void setModelPath(String path) {
        this.path = path;
    }

    public final String getModelPath() {
        return path;
    }

    private CvSVM svm = new CvSVM();

    // ! EasyPR的getFeatures回调函数
    // ！用于从车牌的image生成svm的训练特征features
    private SVMCallback features = new Features();

    //! 模型存储路径
    private String path = "res/model/svm.xml";
}
