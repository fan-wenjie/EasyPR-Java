package org.easypr.core;

import static org.bytedeco.javacpp.opencv_core.*;

/*
 * Created by fanwenjie
 * @version 1.1
 */

public class CharsRecognise {


    public void loadANN(String s)
    {
        charsIdentify.loadModel(s);
    }

    public String charsRecognise(Mat plate)
    {
        //车牌字符方块集合
        MatVector matVec = new MatVector();

        String plateIdentify = "";

        int result = charsSegment.charsSegment(plate, matVec);
        if (result == 0)
        {
            int num =(int) matVec.size();
            for (int j = 0; j < num; j++)
            {
                Mat charMat = matVec.get(j);
                boolean isChinses = false;

                //默认首个字符块是中文字符
                if (j == 0)
                    isChinses = true;

                String charcater = charsIdentify.charsIdentify(charMat, isChinses);
                plateIdentify = plateIdentify + charcater;
            }
        }
        return plateIdentify;
    }

    //! 是否开启调试模式
    public void setCRDebug(boolean isDebug){ charsSegment.setDebug(isDebug);}

    //! 获取调试模式状态
    public boolean getCRDebug(){ return charsSegment.getDebug();}


    //! 获得车牌颜色
    public final String getPlateType(Mat input)
    {
        String color = "未知";
        int result = charsSegment.getPlateType(input);
        if (1 == result)
            color = "蓝牌";
        if (2 == result)
            color = "黄牌";
        return color;
    }

    //! 设置变量
    public void setLiuDingSize(int param){ charsSegment.setLiuDingSize(param);}
    public void setColorThreshold(int param){ charsSegment.setColorThreshold(param);}
    public void setBluePercent(float param){ charsSegment.setBluePercent(param);}
    public final float getBluePercent() { return charsSegment.getBluePercent();}
    public void setWhitePercent(float param){ charsSegment.setWhitePercent(param);}
    public final float getWhitePercent()  { return charsSegment.getWhitePercent();}

    //！字符分割
    private CharsSegment charsSegment = new CharsSegment();

    //! 字符识别
    private CharsIdentify charsIdentify = new CharsIdentify();
}
