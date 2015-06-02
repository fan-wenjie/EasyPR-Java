package org.easypr.core;

import org.bytedeco.javacpp.opencv_core.Mat;


/**
 * @author Created by fanwenjie
 * @author lin.yao
 *
 */
public interface SVMCallback {

    /***
     * EasyPR的getFeatures回调函数,本函数是生成直方图均衡特征的回调函数
     * 
     * @param image
     * @return
     */
    public abstract Mat getHisteqFeatures(final Mat image);

    /**
     * EasyPR的getFeatures回调函数, 本函数是获取垂直和水平的直方图图值
     * 
     * @param image
     * @return
     */
    public abstract Mat getHistogramFeatures(final Mat image);

    /**
     * 本函数是获取SITF特征子的回调函数
     * 
     * @param image
     * @return
     */
    public abstract Mat getSIFTFeatures(final Mat image);

    /**
     * 本函数是获取HOG特征子的回调函数
     * 
     * @param image
     * @return
     */
    public abstract Mat getHOGFeatures(final Mat image);
}
