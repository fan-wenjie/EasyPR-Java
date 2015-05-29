package org.easypr.core;

import static org.bytedeco.javacpp.opencv_highgui.imwrite;

import java.util.Vector;

import org.bytedeco.javacpp.opencv_core.Mat;

/**
 * @author lin.yao
 * 
 */
public class PlateDetect {

    /**
     * 可能是车牌的图块集合
     * 
     * @param src
     * @param resultVec
     * @return
     */
    public int plateDetect(final Mat src, Vector<Mat> resultVec) {
        Vector<Mat> matVec = plateLocate.plateLocate(src);
        if (0 == matVec.size())
            return -1;
        int resultJu = plateJudge.plateJudge(matVec, resultVec);

        if (getPDDebug()) {
            int size = (int) resultVec.size();
            for (int i = 0; i < size; i++) {
                Mat img = resultVec.get(i);
                String str = "image/tmp/plate_judge_result_"
                        + Integer.valueOf(i).toString() + ".jpg";
                imwrite(str, img);
            }
        }
        if (0 != resultJu)
            return -2;
        return 0;
    }

    /**
     * 生活模式与工业模式切换
     * 
     * @param pdLifemode
     */
    public void setPDLifemode(boolean pdLifemode) {
        plateLocate.setLifemode(pdLifemode);
    }

    // ! 是否开启调试模式
    public void setPDDebug(boolean pdDebug) {
        plateLocate.setDebug(pdDebug);
    }

    // ! 获取调试模式状态
    public boolean getPDDebug() {
        return plateLocate.getDebug();
    }

    // ! 设置与读取变量
    public void setGaussianBlurSize(int gaussianBlurSize) {
        plateLocate.setGaussianBlurSize(gaussianBlurSize);
    }

    public final int getGaussianBlurSize() {
        return plateLocate.getGaussianBlurSize();
    }

    public void setMorphSizeWidth(int morphSizeWidth) {
        plateLocate.setMorphSizeWidth(morphSizeWidth);
    }

    public final int getMorphSizeWidth() {
        return plateLocate.getMorphSizeWidth();
    }

    public void setMorphSizeHeight(int morphSizeHeight) {
        plateLocate.setMorphSizeHeight(morphSizeHeight);
    }

    public final int getMorphSizeHeight() {
        return plateLocate.getMorphSizeHeight();
    }

    public void setVerifyError(float verifyError) {
        plateLocate.setVerifyError(verifyError);
    }

    public final float getVerifyError() {
        return plateLocate.getVerifyError();
    }

    public void setVerifyAspect(float verifyAspect) {
        plateLocate.setVerifyAspect(verifyAspect);
    }

    public final float getVerifyAspect() {
        return plateLocate.getVerifyAspect();
    }

    public void setVerifyMin(int verifyMin) {
        plateLocate.setVerifyMin(verifyMin);
    }

    public void setVerifyMax(int verifyMax) {
        plateLocate.setVerifyMax(verifyMax);
    }

    public void setJudgeAngle(int judgeAngle) {
        plateLocate.setJudgeAngle(judgeAngle);
    }

    // ！车牌定位
    private PlateLocate plateLocate = new PlateLocate();

    // ! 车牌判断
    private PlateJudge plateJudge = new PlateJudge();

}
