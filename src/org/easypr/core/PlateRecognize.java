package org.easypr.core;

import org.bytedeco.javacpp.opencv_core.*;

import java.util.Vector;

/**
 * Created by Anonymous on 2015-03-17.
 */
public class PlateRecognize {

    public int plateRecognize(Mat src, Vector<String> licenseVec) {
        //车牌方块集合
        MatVector plateVec = new MatVector();
        int resultPD = plateDetect.plateDetect(src, plateVec);
        if (resultPD == 0) {
            int num = (int) plateVec.size();
            for (int j = 0; j < num; j++) {
                Mat plate = plateVec.get(j);
                //获取车牌颜色
                String plateType = charsRecognise.getPlateType(plate);
                //获取车牌号
                String plateIdentify = charsRecognise.charsRecognise(plate);
                String license = plateType + ":" + plateIdentify;
                licenseVec.add(license);
            }
        }
        return resultPD;
    }

    public void setLifemode(boolean lifemode) {
        plateDetect.setPDLifemode(lifemode);
    }

    //! 是否开启调试模式
    public void setDebug(boolean debug) {
        plateDetect.setPDDebug(debug);
        charsRecognise.setCRDebug(debug);
    }

    private PlateDetect plateDetect = new PlateDetect();
    private CharsRecognise charsRecognise = new CharsRecognise();
}
