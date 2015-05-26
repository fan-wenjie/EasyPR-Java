package org.easypr.core;

import static org.bytedeco.javacpp.opencv_core.countNonZero;
import static org.bytedeco.javacpp.opencv_core.merge;
import static org.bytedeco.javacpp.opencv_core.split;
import static org.bytedeco.javacpp.opencv_highgui.cvShowImage;
import static org.bytedeco.javacpp.opencv_highgui.cvWaitKey;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2HSV;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.equalizeHist;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;

/**
 * @author lin.yao
 * 
 */
public class CoreFunc {
    public enum Color {
        BLUE, YELLOW, UNKNOWN
    };

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
                int H = p.get(j);
                int S = p.get(j + 1);
                int V = p.get(j + 2);

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

                    if ((S > min_sv && S < max_sv) && (V > min_sv && V < max_sv))
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
        final float thresh = 0.2f;

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
