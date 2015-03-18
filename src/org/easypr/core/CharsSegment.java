package org.easypr.core;

import org.bytedeco.javacpp.BytePointer;
import org.easypr.util.Convert;

import java.util.Vector;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_highgui.*;

/*
 * Created by fanwenjie
 * @version 1.1
 */

public class CharsSegment {
    final static float DEFAULT_BLUEPERCEMT = 0.3f;
    final static float DEFAULT_WHITEPERCEMT = 0.1f;

    private int liuDingSize;
    private int theMatWidth;
    private int colorThreshold;
    private float bluePercent;
    private float whitePercent;
    private boolean isDebug;

    public static final int DEFAULT_LIUDING_SIZE = 7;
    public static final int DEFAULT_MAT_WIDTH = 136;
    public static final int DEFAULT_COLORTHRESHOLD = 150;
    //! 是否开启调试模式常量，默认0代表关闭
    public static final boolean DEFAULT_DEBUG = false;

    //! preprocessChar所用常量
    public static final int CHAR_SIZE = 20;
    public static final int HORIZONTAL = 1;
    public static final int VERTICAL = 0;


    public CharsSegment() {
        this.liuDingSize = DEFAULT_LIUDING_SIZE;
        this.theMatWidth = DEFAULT_MAT_WIDTH;

        //！车牌颜色判断参数
        this.colorThreshold = DEFAULT_COLORTHRESHOLD;
        this.bluePercent = DEFAULT_BLUEPERCEMT;
        this.whitePercent = DEFAULT_WHITEPERCEMT;

        this.isDebug = DEFAULT_DEBUG;
    }

    //! 字符分割
    public int charsSegment(Mat input, MatVector resultVec) {
        if (input.data().isNull())
            return -3;

        //判断车牌颜色以此确认threshold方法
        int plateType = getPlateType(input);
        cvtColor(input, input, CV_RGB2GRAY);

        //Threshold input image
        Mat img_threshold = new Mat();
        if (1 == plateType)
            threshold(input, img_threshold, 10, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
        else
            threshold(input, img_threshold, 10, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);

        if (this.isDebug) {
            String str = "image/tmp/debug_char_threshold.jpg";
            imwrite(str, img_threshold);
        }

        //去除车牌上方的柳钉以及下方的横线等干扰
        clearLiuDing(img_threshold);


        if (this.isDebug) {
            String str = "image/tmp/debug_char_clearLiuDing.jpg";
            imwrite(str, img_threshold);
        }

        Mat img_contours = new Mat();
        img_threshold.copyTo(img_contours);

        MatVector contours = new MatVector();

        findContours(img_contours,
                contours, // a vector of contours
                CV_RETR_EXTERNAL, // retrieve the external contours
                CV_CHAIN_APPROX_NONE); // all pixels of each contours

        //Start to iterate to each contour founded


        //Remove patch that are no inside limits of aspect ratio and area.
        //将不符合特定尺寸的图块排除出去
        Vector<Rect> vecRect = new Vector<Rect>();
        for (int i = 0; i < contours.size(); ++i) {
            Rect mr = boundingRect(contours.get(i));
            if (verifySizes(new Mat(img_threshold, mr)))
                vecRect.add(mr);
        }


        if (vecRect.size() == 0)
            return -3;

        Vector<Rect> sortedRect = new Vector<Rect>();
        //对符合尺寸的图块按照从左到右进行排序
        SortRect(vecRect, sortedRect);

        int specIndex = 0;
        //获得指示城市的特定Rect,如苏A的"A"
        specIndex = GetSpecificRect(sortedRect);

        if (this.isDebug) {
            if (specIndex < sortedRect.size()) {
                Mat specMat = new Mat(img_threshold, sortedRect.get(specIndex));
                String str = "image/tmp/debug_specMat.jpg";
                imwrite(str, specMat);
            }
        }

        //根据特定Rect向左反推出中文字符
        //这样做的主要原因是根据findContours方法很难捕捉到中文字符的准确Rect，因此仅能
        //退过特定算法来指定
        Rect chineseRect = new Rect();
        if (specIndex < sortedRect.size())
            chineseRect = GetChineseRect(sortedRect.get(specIndex));
        else
            return -3;

        if (this.isDebug) {
            Mat chineseMat = new Mat(img_threshold, chineseRect);
            String str = "image/tmp/debug_chineseMat.jpg";
            imwrite(str, chineseMat);
        }


        //新建一个全新的排序Rect
        //将中文字符Rect第一个加进来，因为它肯定是最左边的
        //其余的Rect只按照顺序去6个，车牌只可能是7个字符！这样可以避免阴影导致的“1”字符
        Vector<Rect> newSortedRect = new Vector<Rect>();
        newSortedRect.add(chineseRect);
        RebuildRect(sortedRect, newSortedRect, specIndex);

        if (newSortedRect.size() == 0)
            return -3;

        for (int i = 0; i < newSortedRect.size(); i++) {
            Rect mr = newSortedRect.get(i);
            Mat auxRoi = new Mat(img_threshold, mr);

            auxRoi = preprocessChar(auxRoi);
            if (this.isDebug) {
                String str = "image/tmp/debug_char_auxRoi_" + Integer.valueOf(i).toString() + ".jpg";
                imwrite(str, auxRoi);
            }
            resultVec.put(auxRoi);
        }
        return 0;
    }

    //! 字符尺寸验证
    public Boolean verifySizes(Mat r) {
        float aspect = 45.0f / 90.0f;
        float charAspect = (float) r.cols() / (float) r.rows();
        float error = 0.7f;
        float minHeight = 10f;
        float maxHeight = 35f;
        //We have a different aspect ratio for number 1, and it can be ~0.2
        float minAspect = 0.05f;
        float maxAspect = aspect + aspect * error;
        //area of pixels
        float area = countNonZero(r);
        //bb area
        float bbArea = r.cols() * r.rows();
        //% of pixel in area
        float percPixels = area / bbArea;

        return  percPixels <= 1 && charAspect > minAspect && charAspect < maxAspect && r.rows() >= minHeight && r.rows() < maxHeight;
    }

    //! 字符预处理
    public Mat preprocessChar(Mat in) {
        //Remap image
        int h = in.rows();
        int w = in.cols();
        int charSize = CHAR_SIZE;    //统一每个字符的大小
        Mat transformMat = Mat.eye(2, 3, CV_32F).asMat();
        int m = (w > h) ? w : h;
        transformMat.ptr(0,2).put(Convert.getBytes(((m-w) / 2f)));
        transformMat.ptr(1,2).put(Convert.getBytes((m-h)/2f));
        Mat warpImage = new Mat(m, m, in.type());
        warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, new Scalar(0));
        Mat out = new Mat();
        resize(warpImage, out, new Size(charSize, charSize));
        return out;
    }
/*
    //! 生成直方图
    public Mat ProjectedHistogram(Mat img, int t) {
        return null;
    }

    //! 生成字符的特定特征
    public Mat features(Mat in, int sizeData) {
        return null;
    }*/

    //! 直方图均衡，为判断车牌颜色做准备
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

    //! 获得车牌颜色
    public int getPlateType(Mat input) {
        Mat img = new Mat();
        input.copyTo(img);
        img = histeq(img);

        double countBlue = 0;
        double countWhite = 0;

        int nums = img.rows() * img.cols();
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                BytePointer pointer = img.ptr(i, j);
                int blue = pointer.get(0) & 0xFF;
                int green = pointer.get(1) & 0xFF;
                int red = pointer.get(2) & 0xFF;

                if (blue > this.colorThreshold && green > 10 && red > 10)
                    countBlue++;
                if (blue > this.colorThreshold && green > this.colorThreshold && red > this.colorThreshold)
                    countWhite++;
            }
        }

        double percentBlue = countBlue / nums;
        double percentWhite = countWhite / nums;

        if (percentBlue - this.bluePercent > 0 && percentWhite - this.whitePercent > 0)
            return 1;
        else
            return 2;
    }

    //! 去除影响字符识别的柳钉
    public Mat clearLiuDing(Mat img) {
        final int x = this.liuDingSize;
        Mat jump = Mat.zeros(1, img.rows(), CV_32F).asMat();
        for (int i = 0; i < img.rows(); i++) {
            int jumpCount = 0;
            for (int j = 0; j < img.cols() - 1; j++) {
                if (img.ptr(i, j).get() != img.ptr(i, j + 1).get())
                    jumpCount++;
            }
            jump.ptr(i).put(Convert.getBytes((float)jumpCount));
        }
        for (int i = 0; i < img.rows(); i++) {

            if (Convert.toFloat(jump.ptr(i))<=x){
                for (int j = 0; j < img.cols(); j++) {
                    img.ptr(i, j).put((byte) 0);
                }
            }
        }
        return img;
    }


    //! 根据特殊车牌来构造猜测中文字符的位置和大小
    public Rect GetChineseRect(final Rect rectSpe) {
        int height = rectSpe.height();
        float newwidth = rectSpe.width() * 1.15f;
        int x = rectSpe.x();
        int y = rectSpe.y();

        int newx = x - (int) (newwidth * 1.15);
        newx = newx > 0 ? newx : 0;
        Rect a = new Rect(newx, y, (int) newwidth, height);
        return a;
    }

    //! 找出指示城市的字符的Rect，例如苏A7003X，就是A的位置
    public int GetSpecificRect(final Vector<Rect> vecRect) {
        Vector<Integer> xpositions = new Vector<Integer>();
        int maxHeight = 0;
        int maxWidth = 0;
        for (int i = 0; i < vecRect.size(); i++) {
            xpositions.add(vecRect.get(i).x());

            if (vecRect.get(i).height() > maxHeight) {
                maxHeight = vecRect.get(i).height();
            }
            if (vecRect.get(i).width() > maxWidth) {
                maxWidth = vecRect.get(i).width();
            }
        }

        int specIndex = 0;
        for (int i = 0; i < vecRect.size(); i++) {
            Rect mr = vecRect.get(i);
            int midx = mr.x() + mr.width() / 2;

            //如果一个字符有一定的大小，并且在整个车牌的1/7到2/7之间，则是我们要找的特殊车牌
            if ((mr.width() > maxWidth * 0.8 || mr.height() > maxHeight * 0.8) &&
                    (midx < this.theMatWidth * 2 / 7 && midx > this.theMatWidth / 7)) {
                specIndex = i;
            }
        }

        return specIndex;
    }

    //! 这个函数做两个事情
    //  1.把特殊字符Rect左边的全部Rect去掉，后面再重建中文字符的位置。
    //  2.从特殊字符Rect开始，依次选择6个Rect，多余的舍去。
    public int RebuildRect(final Vector<Rect> vecRect, Vector<Rect> outRect, int specIndex) {
        //最大只能有7个Rect,减去中文的就只有6个Rect
        int count = 6;
        for (int i = 0; i < vecRect.size(); i++) {
            //将特殊字符左边的Rect去掉，这个可能会去掉中文Rect，不过没关系，我们后面会重建。
            if (i < specIndex)
                continue;

            outRect.add(vecRect.get(i));
            if (--count == 0)
                break;
        }
        return 0;
    }

    //! 将Rect按位置从左到右进行排序
    int SortRect(final Vector<Rect> vecRect, Vector<Rect> out) {
        Vector<Integer> orderIndex = new Vector<Integer>();
        Vector<Integer> xpositions = new Vector<Integer>();
        for (int i = 0; i < vecRect.size(); ++i) {
            orderIndex.add(i);
            xpositions.add(vecRect.get(i).x());
        }

        float min = xpositions.get(0);
        int minIdx;
        for (int i = 0; i < xpositions.size(); ++i) {
            min = xpositions.get(i);
            minIdx = i;
            for (int j = i; i < xpositions.size(); ++j) {
                if (xpositions.get(j) < min) {
                    min = xpositions.get(j);
                    minIdx = j;
                }
            }
            int aux_i = orderIndex.get(i);
            int aux_min = orderIndex.get(minIdx);
            orderIndex.insertElementAt(aux_min, i);
            orderIndex.insertElementAt(aux_i, minIdx);

            float aux_xi = xpositions.get(i);
            float aux_xmin = xpositions.get(minIdx);
            xpositions.insertElementAt((int) aux_xmin, i);
            xpositions.insertElementAt((int) aux_xi, minIdx);
        }

        for (int i = 0; i < orderIndex.size(); i++)
            out.add(vecRect.get(orderIndex.get(i)));
        return 0;
    }

    //! 设置变量
    public void setLiuDingSize(int param) {
        this.liuDingSize = param;
    }

    public void setColorThreshold(int param) {
        this.colorThreshold = param;
    }

    public void setBluePercent(float param) {
        this.bluePercent = param;
    }

    public final float getBluePercent() {
        return this.bluePercent;
    }

    public void setWhitePercent(float param) {
        this.whitePercent = param;
    }

    public final float getWhitePercent() {
        return this.whitePercent;
    }

    public boolean getDebug() {
        return this.isDebug;
    }

    public void setDebug(boolean isDebug) {
        this.isDebug = isDebug;
    }
}
