package org.easypr.core;

import org.easypr.util.Convert;

import static org.bytedeco.javacpp.opencv_ml.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

import static org.bytedeco.javacpp.opencv_core.*;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Created by fanwenjie
 * @author lin.yao
 * 
 */
public class CharsIdentify {

    public CharsIdentify() {
        loadModel();
        
        if (this.map.isEmpty()) {
            map.put("zh_cuan","川");
            map.put("zh_e","鄂");
            map.put("zh_gan","赣");
            map.put("zh_gan1","甘");
            map.put("zh_gui","贵");
            map.put("zh_gui1","桂");
            map.put("zh_hei","黑");
            map.put("zh_hu","沪");
            map.put("zh_ji","冀");
            map.put("zh_jin","津");
            map.put("zh_jing","京");
            map.put("zh_jl","吉");
            map.put("zh_liao","辽");
            map.put("zh_lu","鲁");
            map.put("zh_meng","蒙");
            map.put("zh_min","闽");
            map.put("zh_ning","宁");
            map.put("zh_qing","青");
            map.put("zh_qiong","琼");
            map.put("zh_shan","陕");
            map.put("zh_su","苏");
            map.put("zh_sx","晋");
            map.put("zh_wan","皖");
            map.put("zh_xiang","湘");
            map.put("zh_xin","新");
            map.put("zh_yu","豫");
            map.put("zh_yu1","渝");
            map.put("zh_yue","粤");
            map.put("zh_yun","云");
            map.put("zh_zang","藏");
            map.put("zh_zhe","浙");
        }
    }

    /**
     * @param input
     * @param isChinese
     * @return
     */
    public String charsIdentify(final Mat input, final Boolean isChinese, final Boolean isSpeci) {
        String result = "";
        
        Mat f = features(input, this.predictSize);
        
        int index = classify(f, isChinese, isSpeci);
        
        if (!isChinese) {
            result = String.valueOf(strCharacters[index]);
            
            System.out.println("strCharacters = " + result); //TODO
        } else {            
            String s = strChinese[index - numCharacter];
            result = map.get(s);
            
            System.out.println("strChinese = " + result); //TODO
        }
        return result;
    }

    private int classify(final Mat f, final Boolean isChinses, final Boolean isSpeci) {
        int result = -1;
        Mat output = new Mat(1, numAll, CV_32FC1);
        
        ann.predict(f, output);
        
        int ann_min = (!isChinses) ? ((isSpeci) ? 10 : 0) : numCharacter;
        int ann_max = (!isChinses) ? numCharacter : numAll;
        
        float maxVal = -2;
        
        for (int j = ann_min; j < ann_max; j++) {
            float val = Convert.toFloat(output.ptr(j));
            
//            System.out.println("output.ptr: " + output.ptr(j).get());
//            System.out.println("val: " + val);
            
            if (val > maxVal) {
                maxVal = val;
                result = j;
            }
        }
        
        return result;
    }

    public float[][] projectedHistogram(final Mat img) {
        float[] nonZeroHor = new float[img.rows()];
        float[] nonZeroVer = new float[img.cols()];
        for (int i = 0; i < img.rows(); ++i)
            for (int j = 0; j < img.cols(); ++j)
                if (0 != img.ptr(i, j).get()) {
                    ++nonZeroHor[i];
                    ++nonZeroVer[j];
                }
        float[][] out = new float[][] { nonZeroVer, nonZeroHor };
        for (int i = 0; i < 2; ++i) {
            float max = 0;
            for (int j = 0; j < out[i].length; ++j)
                if (max < out[i][j])
                    max = out[i][j];
            if (max > 0)
                for (int j = 0; j < out[i].length; ++j)
                    out[i][j] /= max;
        }
        return out;
    }

    private Mat features(final Mat in, int sizeData) {

        float[][] hist = projectedHistogram(in);
        float[] vhist = hist[0];
        float[] hhist = hist[1];

        Mat lowData = new Mat();
        resize(in, lowData, new Size(sizeData, sizeData));

        int numCols = vhist.length + hhist.length + lowData.cols() * lowData.rows();
        Mat out = Mat.zeros(1, numCols, CV_32F).asMat();

        int j = 0;
        for (int i = 0; i < vhist.length; ++i, ++j) {
            out.ptr(j).put(Convert.getBytes(vhist[i]));
        }
        for (int i = 0; i < hhist.length; ++i, ++j) {
            out.ptr(j).put(Convert.getBytes(hhist[i]));
        }
        for (int x = 0; x < lowData.cols(); x++)
            for (int y = 0; y < lowData.rows(); y++, ++j) {
                float val = lowData.ptr(x, y).get() & 0xFF;
                out.ptr(j).put(Convert.getBytes(val));
            }
        return out;
    }

    private void loadModel() {
        loadModel(this.path);
    }

    public void loadModel(String s) {
        this.ann.clear();
        this.ann.load(s, "ann");
    }

    static boolean hasPrint = false;

    public final void setModelPath(String path) {
        this.path = path;
    }

    public final String getModelPath() {
        return this.path;
    }

    private CvANN_MLP ann = new CvANN_MLP();

    private String path = "res/model/ann.xml";

    private int predictSize = 10;

    private Map<String, String> map = new HashMap<String, String>();

    private final char strCharacters[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
            'F', 'G', 'H', /* 没有I */'J', 'K', 'L', 'M', 'N', /* 没有O */'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
            'Z' };
    private final static int numCharacter = 34; // 没有I和0,10个数字与24个英文字符之和

    private final String strChinese[] = { "zh_cuan" /* 川 */, "zh_e" /* 鄂 */, "zh_gan" /* 赣 */, "zh_gan1"/* 甘 */,
            "zh_gui"/* 贵 */, "zh_gui1"/* 桂 */, "zh_hei" /* 黑 */, "zh_hu" /* 沪 */, "zh_ji" /* 冀 */, "zh_jin" /* 津 */,
            "zh_jing" /* 京 */, "zh_jl" /* 吉 */, "zh_liao" /* 辽 */, "zh_lu" /* 鲁 */, "zh_meng" /* 蒙 */,
            "zh_min" /* 闽 */, "zh_ning" /* 宁 */, "zh_qing" /* 青 */, "zh_qiong" /* 琼 */, "zh_shan" /* 陕 */,
            "zh_su" /* 苏 */, "zh_sx" /* 晋 */, "zh_wan" /* 皖 */, "zh_xiang" /* 湘 */, "zh_xin" /* 新 */, "zh_yu" /* 豫 */,
            "zh_yu1" /* 渝 */, "zh_yue" /* 粤 */, "zh_yun" /* 云 */, "zh_zang" /* 藏 */, "zh_zhe" /* 浙 */};
    private final static int numChinese = 31;
    
    private final static int numAll = 65; /* 34+31=65 */
    
    private final static int HORIZONTAL = 1;
    private final static int VERTICAL = 0;
}
