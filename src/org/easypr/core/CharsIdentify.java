package org.easypr.core;

import org.easypr.util.MatHelper;

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
        this.predictSize = 20;
        this.path = "./model/ann.xml";
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
        Mat output = new Mat(1,numAll,CV_32F);
        ann.predict(f,output);
        if(!isChinses)
        {
            result = 0;
            float maxVal = -2;
            for(int j=0;j<numCharacter;j++){
                float val = (Float) MatHelper.getElement(output, j);
                if(val > maxVal){
                    maxVal = val;
                    result = j;
                }
            }
        }else{
            result = numCharacter;
            float maxVal = -2;
            for(int j=numCharacter;j<numAll;++j){
                float val = (Float)MatHelper.getElement(output,j);
                if(val > maxVal){
                    maxVal = val;
                    result = j;
                }
            }
        }
        return result;
    }

    public Mat projectedHistogram(final Mat img,int t){
        int sz = (t==0)?img.cols():img.rows();
        Mat mhist = Mat.zeros(1,sz,CV_32F).asMat();
        for(int j=0; j<sz; j++){
            Mat data = (t==0)?img.col(j):img.row(j);
            MatHelper.setElement(mhist,(float)countNonZero(data),j);
        }
        float max = 0;
        for(int j=0;j<sz;++j)
            if((Float)MatHelper.getElement(mhist,j)>max)
                max = (Float)MatHelper.getElement(mhist,j);
        if(max>0)
            mhist.convertTo(mhist,-1,1.0f/max,0);
        return mhist;
    }

    public Mat features(final Mat in,int sizeData){

        Mat vhist = projectedHistogram(in,VERTICAL);
        Mat hhist = projectedHistogram(in,HORIZONTAL);

        Mat lowData = new Mat();
        resize(in,lowData,new Size(sizeData,sizeData));

        int numCols = vhist.cols() + hhist.cols() + lowData.cols()*lowData.rows();
        Mat out = Mat.zeros(1,numCols,CV_32F).asMat();

        int j = 0;
        for(int i =0;i<vhist.cols();++i,++j){
            float val = (Float)MatHelper.getElement(vhist,i);
            MatHelper.setElement(out,val,j);
        }
        for(int i =0;i<hhist.cols();++i,++j){
            float val = (Float)MatHelper.getElement(hhist,i);
            MatHelper.setElement(out,val,j);
        }
        for(int x=0; x<lowData.cols(); x++)
            for(int y=0; y<lowData.rows(); y++,++j){
                float val = lowData.ptr(x,y).get()&0xFF;
                MatHelper.setElement(out,val,j);
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

    public final void setModelPath(String path){
        this.path = path;
    }

    public final String getModelPath(){
        return this.path;
    }

    private CvANN_MLP ann = new CvANN_MLP();

    private String path;

    private int predictSize;

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
