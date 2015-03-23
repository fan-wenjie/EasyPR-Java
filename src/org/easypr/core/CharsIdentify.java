package org.easypr.core;

import org.easypr.util.Convert;

import static org.bytedeco.javacpp.opencv_ml.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

import static org.bytedeco.javacpp.opencv_core.*;


import java.util.HashMap;
import java.util.Map;

/*
 * Created by fanwenjie
 * @version 1.1
 */

public class CharsIdentify {

    public CharsIdentify(){
        loadModel();
        if(this.map.isEmpty()){
            this.map.put("zh_cuan","川");
            this.map.put("zh_e","鄂");
            this.map.put("zh_gan","赣");
            this.map.put("zh_hei","黑");
            this.map.put("zh_hu","沪");
            this.map.put("zh_ji","冀");
            this.map.put("zh_jl","吉");
            this.map.put("zh_jin","津");
            this.map.put("zh_jing","京");
            this.map.put("zh_shan","陕");
            this.map.put("zh_liao","辽");
            this.map.put("zh_lu","鲁");
            this.map.put("zh_min","闽");
            this.map.put("zh_ning","宁");
            this.map.put("zh_su","苏");
            this.map.put("zh_sx","晋");
            this.map.put("zh_wan","皖");
            this.map.put("zh_yu","豫");
            this.map.put("zh_yue","粤");
            this.map.put("zh_zhe","浙");
        }
    }

    public String charsIdentify(Mat input,Boolean isChinese){
        Mat f = features(input,this.predictSize);
        String result = "";
        int index = classify(f,isChinese);
        if(!isChinese)
            result = result + strCharacters[index];
        else{
            String s = strChinese[index - numCharacter];
            String province = map.get(s);
            result = province + result;
        }
        return result;
    }

    public int classify(Mat f, Boolean isChinses){
        int result;
        Mat output = new Mat(1,numAll,CV_32FC1);

        if(!hasPrint)
        System.out.println("-----------");
        if(!hasPrint)
            for(int i=0;i<f.size(1);++i)
                System.out.println(Convert.toFloat(f.ptr(i)));
        hasPrint = true;
        ann.predict(f,output);
        if(!isChinses)
        {
            result = 0;
            float maxVal = -2;
            for(int j=0;j<numCharacter;j++){
                float val = Convert.toFloat(output.ptr(j));
                if(val > maxVal){
                    maxVal = val;
                    result = j;
                }
            }
        }else{
            result = numCharacter;
            float maxVal = -2;
            for(int j=numCharacter;j<numAll;++j){
                float val = Convert.toFloat(output.ptr(j));
                if(val > maxVal){
                    maxVal = val;
                    result = j;
                }
            }
        }
        return result;
    }

    public float[][] projectedHistogram(final Mat img){
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

    public Mat features(final Mat in,int sizeData){

        float [][]hist = projectedHistogram(in);
        float[] vhist = hist[0];
        float[] hhist = hist[1];

        Mat lowData = new Mat();
        resize(in,lowData,new Size(sizeData,sizeData));

        int numCols = vhist.length + hhist.length + lowData.cols()*lowData.rows();
        Mat out = Mat.zeros(1,numCols,CV_32F).asMat();

        int j = 0;
        for(int i =0;i<vhist.length;++i,++j){
            out.ptr(j).put(Convert.getBytes(vhist[i]));
        }
        for(int i =0;i<hhist.length;++i,++j){
            out.ptr(j).put(Convert.getBytes(hhist[i]));
        }
        for(int x=0; x<lowData.cols(); x++)
            for(int y=0; y<lowData.rows(); y++,++j){
                float val = lowData.ptr(x,y).get()&0xFF;
                out.ptr(j).put(Convert.getBytes(val));
            }
        return out;
    }

    public void loadModel(){
        loadModel(this.path);
    }

    public void loadModel(String s){
        this.ann.clear();
        this.ann.load(s,"ann");
    }

    static boolean hasPrint = false;

    public final void setModelPath(String path){
        this.path = path;
    }

    public final String getModelPath(){
        return this.path;
    }

    private CvANN_MLP ann = new CvANN_MLP();

    private String path = "res/model/ann.xml";

    private int predictSize = 10;

    private Map<String,String> map = new HashMap<String, String>();

    private final char strCharacters[] = {'0','1','2','3','4','5',
            '6','7','8','9','A','B', 'C', 'D', 'E','F', 'G', 'H', /* 没有I */
            'J', 'K', 'L', 'M', 'N', /* 没有O */ 'P', 'Q', 'R', 'S', 'T',
            'U','V', 'W', 'X', 'Y', 'Z'};

    private final String strChinese[] = {"zh_cuan" /* 川 */, "zh_e" /* 鄂 */,  "zh_gan" /* 赣*/,
            "zh_hei" /* 黑 */, "zh_hu" /* 沪 */,  "zh_ji" /* 冀 */,
            "zh_jl" /* 吉 */, "zh_jin" /* 津 */, "zh_jing" /* 京 */, "zh_shan" /* 陕 */,
            "zh_liao" /* 辽 */, "zh_lu" /* 鲁 */, "zh_min" /* 闽 */, "zh_ning" /* 宁 */,
            "zh_su" /* 苏 */,  "zh_sx" /* 晋 */, "zh_wan" /* 皖 */,
            "zh_yu" /* 豫 */, "zh_yue" /* 粤 */, "zh_zhe" /* 浙 */};


    private final int numAll = 54;

    private final int numCharacter = numAll - strCharacters.length;

    private final static int HORIZONTAL = 1;

    private final static int VERTICAL = 0;
}
