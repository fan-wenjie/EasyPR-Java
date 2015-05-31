package org.easypr.core;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.cvShowImage;
import static org.bytedeco.javacpp.opencv_highgui.cvWaitKey;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2HSV;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.equalizeHist;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.indexer.FloatIndexer;

/**
 * @author lin.yao
 * 
 */
public class CoreFunc {
    public enum Color {
        UNKNOWN, BLUE, YELLOW
    };

    public enum Direction {
        UNKNOWN, VERTICAL, HORIZONTAL
    }

    /**
     * 根据一幅图像与颜色模板获取对应的二值图
     * 
     * @param src
     *            输入RGB图像
     * @param r
     *            颜色模板（蓝色、黄色）
     * @param adaptive_minsv
     *            S和V的最小值由adaptive_minsv这个bool值判断
     *            <ul>
     *            <li>如果为true，则最小值取决于H值，按比例衰减
     *            <li>如果为false，则不再自适应，使用固定的最小值minabs_sv
     *            </ul>
     * @return 输出灰度图（只有0和255两个值，255代表匹配，0代表不匹配）
     */
    public static Mat colorMatch(final Mat src, final Color r, final boolean adaptive_minsv) {
        final float max_sv = 255;
        final float minref_sv = 64;
        final float minabs_sv = 95;

        // blue的H范围
        final int min_blue = 100;
        final int max_blue = 140;

        // yellow的H范围
        final int min_yellow = 15;
        final int max_yellow = 40;

        // 转到HSV空间进行处理，颜色搜索主要使用的是H分量进行蓝色与黄色的匹配工作
        Mat src_hsv = new Mat();
        cvtColor(src, src_hsv, CV_BGR2HSV);
        MatVector hsvSplit = new MatVector();
        split(src_hsv, hsvSplit);
        equalizeHist(hsvSplit.get(2), hsvSplit.get(2));
        merge(hsvSplit, src_hsv);

        // 匹配模板基色,切换以查找想要的基色
        int min_h = 0;
        int max_h = 0;
        switch (r) {
        case BLUE:
            min_h = min_blue;
            max_h = max_blue;
            break;
        case YELLOW:
            min_h = min_yellow;
            max_h = max_yellow;
            break;
        default:
            break;
        }

        float diff_h = (float) ((max_h - min_h) / 2);
        int avg_h = (int) (min_h + diff_h);

        int channels = src_hsv.channels();
        int nRows = src_hsv.rows();
        // 图像数据列需要考虑通道数的影响；
        int nCols = src_hsv.cols() * channels;

        // 连续存储的数据，按一行处理
        if (src_hsv.isContinuous()) {
            nCols *= nRows;
            nRows = 1;
        }

        for (int i = 0; i < nRows; ++i) {
            BytePointer p = src_hsv.ptr(i);
            for (int j = 0; j < nCols; j += 3) {
                int H = p.get(j) & 0xFF;
                int S = p.get(j + 1) & 0xFF;
                int V = p.get(j + 2) & 0xFF;

                boolean colorMatched = false;

                if (H > min_h && H < max_h) {
                    int Hdiff = 0;
                    if (H > avg_h)
                        Hdiff = H - avg_h;
                    else
                        Hdiff = avg_h - H;

                    float Hdiff_p = Hdiff / diff_h;

                    float min_sv = 0;
                    if (true == adaptive_minsv)
                        min_sv = minref_sv - minref_sv / 2 * (1 - Hdiff_p);
                    else
                        min_sv = minabs_sv;

                    if ((S > min_sv && S <= max_sv) && (V > min_sv && V <= max_sv))
                        colorMatched = true;
                }

                if (colorMatched == true) {
                    p.put(j, (byte) 0);
                    p.put(j + 1, (byte) 0);
                    p.put(j + 2, (byte) 255);
                } else {
                    p.put(j, (byte) 0);
                    p.put(j + 1, (byte) 0);
                    p.put(j + 2, (byte) 0);
                }
            }
        }

        // 获取颜色匹配后的二值灰度图
        MatVector hsvSplit_done = new MatVector();
        split(src_hsv, hsvSplit_done);
        Mat src_grey = hsvSplit_done.get(2);

        return src_grey;
    }

    /**
     * 判断一个车牌的颜色
     * 
     * @param src
     *            车牌mat
     * @param r
     *            颜色模板
     * @param adaptive_minsv
     *            S和V的最小值由adaptive_minsv这个bool值判断
     *            <ul>
     *            <li>如果为true，则最小值取决于H值，按比例衰减
     *            <li>如果为false，则不再自适应，使用固定的最小值minabs_sv
     *            </ul>
     * @return
     */
    public static boolean plateColorJudge(final Mat src, final Color color, final boolean adaptive_minsv) {
        // 判断阈值
        final float thresh = 0.49f;

        Mat gray = colorMatch(src, color, adaptive_minsv);

        float percent = (float) countNonZero(gray) / (gray.rows() * gray.cols());

        return (percent > thresh) ? true : false;
    }

    /**
     * getPlateType 判断车牌的类型
     * 
     * @param src
     * @param adaptive_minsv
     *            S和V的最小值由adaptive_minsv这个bool值判断
     *            <ul>
     *            <li>如果为true，则最小值取决于H值，按比例衰减
     *            <li>如果为false，则不再自适应，使用固定的最小值minabs_sv
     *            </ul>
     * @return
     */
    public static Color getPlateType(final Mat src, final boolean adaptive_minsv) {
        if (plateColorJudge(src, Color.BLUE, adaptive_minsv) == true) {
            return Color.BLUE;
        } else if (plateColorJudge(src, Color.YELLOW, adaptive_minsv) == true) {
            return Color.YELLOW;
        } else {
            return Color.UNKNOWN;
        }
    }

    /**
     * 获取垂直或水平方向直方图
     * 
     * @param img
     * @param direction
     * @return
     */
    public static float[] projectedHistogram(final Mat img, Direction direction) {
        int sz = 0;
        switch (direction) {
        case HORIZONTAL:
            sz = img.rows();
            break;

        case VERTICAL:
            sz = img.cols();
            break;

        default:
            break;
        }

        // 统计这一行或一列中，非零元素的个数，并保存到nonZeroMat中
        float[] nonZeroMat = new float[sz];
        extractChannel(img, img, 0);
        for (int j = 0; j < sz; j++) {
            Mat data = (direction == Direction.HORIZONTAL) ? img.row(j) : img.col(j);
            int count = countNonZero(data);
            nonZeroMat[j] = count;
        }

        // Normalize histogram
        float max = 0;
        for (int j = 0; j < nonZeroMat.length; ++j) {
            max = Math.max(max, nonZeroMat[j]);
        }

        if (max > 0) {
            for (int j = 0; j < nonZeroMat.length; ++j) {
                nonZeroMat[j] /= max;
            }
        }

        return nonZeroMat;
    }

    /**
     * Assign values to feature
     * <p>
     * 样本特征为水平、垂直直方图和低分辨率图像所组成的矢量
     * 
     * @param in
     * @param sizeData
     *            低分辨率图像size = sizeData*sizeData, 可以为0
     * @return
     */
    public static Mat features(final Mat in, final int sizeData) {

        float[] vhist = projectedHistogram(in, Direction.VERTICAL);
        float[] hhist = projectedHistogram(in, Direction.HORIZONTAL);

        Mat lowData = new Mat();
        if (sizeData > 0) {
            resize(in, lowData, new Size(sizeData, sizeData));
        }

        int numCols = vhist.length + hhist.length + lowData.cols() * lowData.rows();
        Mat out = Mat.zeros(1, numCols, CV_32F).asMat();
        FloatIndexer idx = out.createIndexer();

        int j = 0;
        for (int i = 0; i < vhist.length; ++i, ++j) {
            idx.put(0, j, vhist[i]);
        }
        for (int i = 0; i < hhist.length; ++i, ++j) {
            idx.put(0, j, hhist[i]);
        }
        for (int x = 0; x < lowData.cols(); x++) {
            for (int y = 0; y < lowData.rows(); y++, ++j) {
                float val = lowData.ptr(x, y).get() & 0xFF;
                idx.put(0, j, val);
            }
        }

        return out;
    }

    /**
     * Show image
     * 
     * @param title
     * @param src
     */
    public static void showImage(final String title, final Mat src) {
        try {
            IplImage image = src.asIplImage();
            if (image != null) {
                cvShowImage(title, image);
                cvWaitKey(0);
            }
        } catch (Exception ex) {
        }
    }
}
