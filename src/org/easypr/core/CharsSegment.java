package org.easypr.core;

import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_core.countNonZero;
import static org.bytedeco.javacpp.opencv_highgui.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.BORDER_CONSTANT;
import static org.bytedeco.javacpp.opencv_imgproc.CV_CHAIN_APPROX_NONE;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RETR_EXTERNAL;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RGB2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_THRESH_BINARY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_THRESH_BINARY_INV;
import static org.bytedeco.javacpp.opencv_imgproc.CV_THRESH_OTSU;
import static org.bytedeco.javacpp.opencv_imgproc.INTER_LINEAR;
import static org.bytedeco.javacpp.opencv_imgproc.boundingRect;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.findContours;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_imgproc.threshold;
import static org.bytedeco.javacpp.opencv_imgproc.warpAffine;
import static org.easypr.core.CoreFunc.getPlateType;

import java.util.Vector;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.easypr.util.Convert;

/**
 * @author lin.yao
 * 
 */
public class CharsSegment {

    /**
     * 字符分割
     * 
     * @param input
     * @param resultVec
     * @return <ul>
     *         <li>more than zero: the number of chars;
     *         <li>-3: null;
     *         </ul>
     */
    public int charsSegment(final Mat input, Vector<Mat> resultVec) {
        if (input.data().isNull())
            return -3;

        // 判断车牌颜色以此确认threshold方法

        Mat img_threshold = new Mat();

        Mat input_grey = new Mat();
        cvtColor(input, input_grey, CV_RGB2GRAY);

        int w = input.cols();
        int h = input.rows();
        Mat tmpMat = new Mat(input, new Rect((int) (w * 0.1), (int) (h * 0.1), (int) (w * 0.8), (int) (h * 0.8)));

        switch (getPlateType(tmpMat, true)) {
        case BLUE:
            threshold(input_grey, img_threshold, 10, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
            break;

        case YELLOW:
            threshold(input_grey, img_threshold, 10, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);
            break;

        default:
            return -3;
        }

        if (this.isDebug) {
            imwrite("tmp/debug_char_threshold.jpg", img_threshold);
        }

        // 去除车牌上方的柳钉以及下方的横线等干扰
        clearLiuDing(img_threshold);

        if (this.isDebug) {
            String str = "tmp/debug_char_clearLiuDing.jpg";
            imwrite(str, img_threshold);
        }

        // 找轮廓
        Mat img_contours = new Mat();
        img_threshold.copyTo(img_contours);

        MatVector contours = new MatVector();

        findContours(img_contours, contours, // a vector of contours
                CV_RETR_EXTERNAL, // retrieve the external contours
                CV_CHAIN_APPROX_NONE); // all pixels of each contours

        // Start to iterate to each contour founded

        // Remove patch that are no inside limits of aspect ratio and area.
        // 将不符合特定尺寸的图块排除出去
        Vector<Rect> vecRect = new Vector<Rect>();
        for (int i = 0; i < contours.size(); ++i) {
            Rect mr = boundingRect(contours.get(i));
            if (verifySizes(new Mat(img_threshold, mr)))
                vecRect.add(mr);
        }

        if (vecRect.size() == 0)
            return -3;

        Vector<Rect> sortedRect = new Vector<Rect>();
        // 对符合尺寸的图块按照从左到右进行排序
        SortRect(vecRect, sortedRect);

        // 获得指示城市的特定Rect,如苏A的"A"
        int specIndex = GetSpecificRect(sortedRect);

        if (this.isDebug) {
            if (specIndex < sortedRect.size()) {
                Mat specMat = new Mat(img_threshold, sortedRect.get(specIndex));
                String str = "tmp/debug_specMat.jpg";
                imwrite(str, specMat);
            }
        }

        // 根据特定Rect向左反推出中文字符
        // 这样做的主要原因是根据findContours方法很难捕捉到中文字符的准确Rect，因此仅能
        // 退过特定算法来指定
        Rect chineseRect = new Rect();
        if (specIndex < sortedRect.size())
            chineseRect = GetChineseRect(sortedRect.get(specIndex));
        else
            return -3;

        if (this.isDebug) {
            Mat chineseMat = new Mat(img_threshold, chineseRect);
            String str = "tmp/debug_chineseMat.jpg";
            imwrite(str, chineseMat);
        }

        // 新建一个全新的排序Rect
        // 将中文字符Rect第一个加进来，因为它肯定是最左边的
        // 其余的Rect只按照顺序去6个，车牌只可能是7个字符！这样可以避免阴影导致的“1”字符
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
                String str = "tmp/debug_char_auxRoi_" + Integer.valueOf(i).toString() + ".jpg";
                imwrite(str, auxRoi);
            }
            resultVec.add(auxRoi);
        }
        return 0;
    }

    /**
     * 字符尺寸验证
     * 
     * @param r
     * @return
     */
    private Boolean verifySizes(Mat r) {
        float aspect = 45.0f / 90.0f;
        float charAspect = (float) r.cols() / (float) r.rows();
        float error = 0.7f;
        float minHeight = 10f;
        float maxHeight = 35f;
        // We have a different aspect ratio for number 1, and it can be ~0.2
        float minAspect = 0.05f;
        float maxAspect = aspect + aspect * error;
        // area of pixels
        float area = countNonZero(r);
        // bb area
        float bbArea = r.cols() * r.rows();
        // % of pixel in area
        float percPixels = area / bbArea;

        return percPixels <= 1 && charAspect > minAspect && charAspect < maxAspect && r.rows() >= minHeight
                && r.rows() < maxHeight;
    }

    /**
     * 字符预处理: 统一每个字符的大小
     * 
     * @param in
     * @return
     */
    private Mat preprocessChar(Mat in) {
        int h = in.rows();
        int w = in.cols();
        int charSize = CHAR_SIZE;
        Mat transformMat = Mat.eye(2, 3, CV_32F).asMat();
        int m = Math.max(w, h);
        transformMat.ptr(0, 2).put(Convert.getBytes(((m - w) / 2f)));
        transformMat.ptr(1, 2).put(Convert.getBytes((m - h) / 2f));

        Mat warpImage = new Mat(m, m, in.type());
        warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, new Scalar(0));

        Mat out = new Mat();
        resize(warpImage, out, new Size(charSize, charSize));

        return out;
    }

    /**
     * 去除车牌上方的钮钉
     * <p>
     * 计算每行元素的阶跃数，如果小于X认为是柳丁，将此行全部填0（涂黑）， X可根据实际调整
     * 
     * @param img
     * @return
     */
    private Mat clearLiuDing(Mat img) {
        final int x = this.liuDingSize;

        Mat jump = Mat.zeros(1, img.rows(), CV_32F).asMat();
        for (int i = 0; i < img.rows(); i++) {
            int jumpCount = 0;
            for (int j = 0; j < img.cols() - 1; j++) {
                if (img.ptr(i, j).get() != img.ptr(i, j + 1).get())
                    jumpCount++;
            }
            jump.ptr(i).put(Convert.getBytes((float) jumpCount));
        }
        for (int i = 0; i < img.rows(); i++) {
            if (Convert.toFloat(jump.ptr(i)) <= x) {
                for (int j = 0; j < img.cols(); j++) {
                    img.ptr(i, j).put((byte) 0);
                }
            }
        }
        return img;
    }

    /**
     * 根据特殊车牌来构造猜测中文字符的位置和大小
     * 
     * @param rectSpe
     * @return
     */
    private Rect GetChineseRect(final Rect rectSpe) {
        int height = rectSpe.height();
        float newwidth = rectSpe.width() * 1.15f;
        int x = rectSpe.x();
        int y = rectSpe.y();

        int newx = x - (int) (newwidth * 1.15);
        newx = Math.max(newx, 0);
        Rect a = new Rect(newx, y, (int) newwidth, height);
        return a;
    }

    /**
     * 找出指示城市的字符的Rect，例如苏A7003X，就是A的位置
     * 
     * @param vecRect
     * @return
     */
    private int GetSpecificRect(final Vector<Rect> vecRect) {
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

            // 如果一个字符有一定的大小，并且在整个车牌的1/7到2/7之间，则是我们要找的特殊车牌
            if ((mr.width() > maxWidth * 0.8 || mr.height() > maxHeight * 0.8)
                    && (midx < this.theMatWidth * 2 / 7 && midx > this.theMatWidth / 7)) {
                specIndex = i;
            }
        }

        return specIndex;
    }

    /**
     * 这个函数做两个事情
     * <ul>
     * <li>把特殊字符Rect左边的全部Rect去掉，后面再重建中文字符的位置;
     * <li>从特殊字符Rect开始，依次选择6个Rect，多余的舍去。
     * <ul>
     * 
     * @param vecRect
     * @param outRect
     * @param specIndex
     * @return
     */
    private int RebuildRect(final Vector<Rect> vecRect, Vector<Rect> outRect, int specIndex) {
        // 最大只能有7个Rect,减去中文的就只有6个Rect
        int count = 6;
        for (int i = 0; i < vecRect.size(); i++) {
            // 将特殊字符左边的Rect去掉，这个可能会去掉中文Rect，不过没关系，我们后面会重建。
            if (i < specIndex)
                continue;

            outRect.add(vecRect.get(i));
            if (--count == 0)
                break;
        }

        return 0;
    }

    /**
     * 将Rect按位置从左到右进行排序
     * 
     * @param vecRect
     * @param out
     * @return
     */
    private void SortRect(final Vector<Rect> vecRect, Vector<Rect> out) {
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
            for (int j = i; j < xpositions.size(); ++j) {
                if (xpositions.get(j) < min) {
                    min = xpositions.get(j);
                    minIdx = j;
                }
            }
            int aux_i = orderIndex.get(i);
            int aux_min = orderIndex.get(minIdx);
            orderIndex.remove(i);
            orderIndex.insertElementAt(aux_min, i);
            orderIndex.remove(minIdx);
            orderIndex.insertElementAt(aux_i, minIdx);

            float aux_xi = xpositions.get(i);
            float aux_xmin = xpositions.get(minIdx);
            xpositions.remove(i);
            xpositions.insertElementAt((int) aux_xmin, i);
            xpositions.remove(minIdx);
            xpositions.insertElementAt((int) aux_xi, minIdx);
        }

        for (int i = 0; i < orderIndex.size(); i++)
            out.add(vecRect.get(orderIndex.get(i)));

        return;
    }

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

    // 是否开启调试模式常量，默认false代表关闭
    final static boolean DEFAULT_DEBUG = false;

    // preprocessChar所用常量
    final static int CHAR_SIZE = 20;
    final static int HORIZONTAL = 1;
    final static int VERTICAL = 0;

    final static int DEFAULT_LIUDING_SIZE = 7;
    final static int DEFAULT_MAT_WIDTH = 136;

    final static int DEFAULT_COLORTHRESHOLD = 150;
    final static float DEFAULT_BLUEPERCEMT = 0.3f;
    final static float DEFAULT_WHITEPERCEMT = 0.1f;

    private int liuDingSize = DEFAULT_LIUDING_SIZE;
    private int theMatWidth = DEFAULT_MAT_WIDTH;

    private int colorThreshold = DEFAULT_COLORTHRESHOLD;
    private float bluePercent = DEFAULT_BLUEPERCEMT;
    private float whitePercent = DEFAULT_WHITEPERCEMT;

    private boolean isDebug = DEFAULT_DEBUG;
}
