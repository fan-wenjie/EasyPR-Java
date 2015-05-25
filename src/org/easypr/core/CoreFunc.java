package org.easypr.core;

import java.util.Vector;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_features2d.*;

/**
 * @author lin.yao
 *
 */
public class CoreFunc {
    public enum Color {
        BLUE, YELLOW
    };

    // ! 根据一幅图像与颜色模板获取对应的二值图
    // ! 输入RGB图像, 颜色模板（蓝色、黄色）
    // ! 输出灰度图（只有0和255两个值，255代表匹配，0代表不匹配）
    public static Mat colorMatch(final Mat src, final Mat match, final Color r, final boolean adaptive_minsv)
    {
        // S和V的最小值由adaptive_minsv这个bool值判断
        // 如果为true，则最小值取决于H值，按比例衰减
        // 如果为false，则不再自适应，使用固定的最小值minabs_sv
        // 默认为false
        final float max_sv = 255;
        final float minref_sv = 64;

        final float minabs_sv = 95;

        //blue的H范围
        final int min_blue = 100;  //100
        final int max_blue = 140;  //140

        //yellow的H范围
        final int min_yellow = 15; //15
        final int max_yellow = 40; //40

        Mat src_hsv;
        // 转到HSV空间进行处理，颜色搜索主要使用的是H分量进行蓝色与黄色的匹配工作
        cvtColor(src, src_hsv, CV_BGR2HSV);

        Vector<Mat> hsvSplit;
        cvSplit(src_hsv, hsvSplit);
        cvEqualizeHist(hsvSplit.get(2), hsvSplit.get(2));
        cvMerge(hsvSplit, src_hsv);

        //匹配模板基色,切换以查找想要的基色
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
        }

        float diff_h = (float)((max_h - min_h) / 2);
        int avg_h = (int) (min_h + diff_h);

        int channels = src_hsv.channels();
        int nRows = src_hsv.rows();
        //图像数据列需要考虑通道数的影响；
        int nCols = src_hsv.cols() * channels;

        if (src_hsv.isContinuous())//连续存储的数据，按一行处理
        {
            nCols *= nRows;
            nRows = 1;
        }

        int i, j;
        uchar* p;
        float s_all = 0;
        float v_all = 0;
        float count = 0;
        for (i = 0; i < nRows; ++i)
        {
            p = src_hsv.ptr<uchar>(i);
            for (j = 0; j < nCols; j += 3)
            {
                int H = int(p[j]); //0-180
                int S = int(p[j + 1]);  //0-255
                int V = int(p[j + 2]);  //0-255

                s_all += S;
                v_all += V;
                count++;

                boolean colorMatched = false;

                if (H > min_h && H < max_h) {
                    int Hdiff = 0;
                    if (H > avg_h)
                        Hdiff = H - avg_h;
                    else
                        Hdiff = avg_h - H;

                    float Hdiff_p = Hdiff / diff_h;

                    // S和V的最小值由adaptive_minsv这个bool值判断
                    // 如果为true，则最小值取决于H值，按比例衰减
                    // 如果为false，则不再自适应，使用固定的最小值minabs_sv
                    float min_sv = 0;
                    if (true == adaptive_minsv)
                        min_sv = minref_sv - minref_sv / 2 * (1 - Hdiff_p); // inref_sv - minref_sv / 2 * (1 - Hdiff_p)
                    else
                        min_sv = minabs_sv; // add

                    if ((S > min_sv && S < max_sv) && (V > min_sv && V < max_sv))
                        colorMatched = true;
                }

                if (colorMatched == true) {
                    p[j] = 0; p[j + 1] = 0; p[j + 2] = 255;
                }
                else {
                    p[j] = 0; p[j + 1] = 0; p[j + 2] = 0;
                }
            }
        }

        // 获取颜色匹配后的二值灰度图
        Mat src_grey;
        Vector<Mat> hsvSplit_done;
        split(src_hsv, hsvSplit_done);
        src_grey = hsvSplit_done.get(2);

        match = src_grey;

        return src_grey;
    }

    // ! 判断一个车牌的颜色
    // ! 输入车牌mat与颜色模板
    // ! 返回true或fasle
    public static boolean plateColorJudge(final Mat src, final Color r, final boolean adaptive_minsv) {
        // 判断阈值
        final float thresh = 0.45f;

        Mat src_gray = new Mat();
        colorMatch(src, src_gray, r, adaptive_minsv);

        float percent = countNonZero(src_gray) / (src_gray.rows() * src_gray.cols());

        if (percent > thresh)
            return true;
        else
            return false;
    }

    // getPlateType
    // 判断车牌的类型
    public static Color getPlateType(final Mat src, final boolean adaptive_minsv) {
        if (plateColorJudge(src, Color.BLUE, adaptive_minsv) == true) {
            return Color.BLUE;
        } else if (plateColorJudge(src, Color.YELLOW, adaptive_minsv) == true) {
            return Color.YELLOW;
        } else {
            return Color.BLUE;
        }
    }
}
