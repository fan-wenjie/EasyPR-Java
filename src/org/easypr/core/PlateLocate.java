package org.easypr.core;

import java.util.Vector;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/*
 * Created by fanwenjie
 * @version 1.1
 */

public class PlateLocate {

    public PlateLocate()
    {
        this.gaussianBlurSize = DEFAULT_GAUSSIANBLUR_SIZE;
        this.morphSizeWidth = DEFAULT_MORPH_SIZE_WIDTH;
        this.morphSizeHeight = DEFAULT_MORPH_SIZE_HEIGHT;

        this.error = DEFAULT_ERROR;
        this.aspect = DEFAULT_ASPECT;
        this.verifyMin = DEFAULT_VERIFY_MIN;
        this.verifyMax = DEFAULT_VERIFY_MAX;

        this.angle = DEFAULT_ANGLE;
        this.debug = DEFAULT_DEBUG;
    }
    //! 生活模式与工业模式切换
//! 如果为真，则设置各项参数为定位生活场景照片（如百度图片）的参数，否则恢复默认值。
    public void setLifemode(boolean islifemode)
    {
        if(islifemode)
        {
            setGaussianBlurSize(5);
            setMorphSizeWidth(9);
            setMorphSizeHeight(3);
            setVerifyError(0.9f);
            setVerifyAspect(4);
            setVerifyMin(1);
            setVerifyMax(30);
        }
        else
        {
            setGaussianBlurSize(DEFAULT_GAUSSIANBLUR_SIZE);
            setMorphSizeWidth(DEFAULT_MORPH_SIZE_WIDTH);
            setMorphSizeHeight(DEFAULT_MORPH_SIZE_HEIGHT);
            setVerifyError(DEFAULT_ERROR);
            setVerifyAspect(DEFAULT_ASPECT);
            setVerifyMin(DEFAULT_VERIFY_MIN);
            setVerifyMax(DEFAULT_VERIFY_MAX);
        }
    }


    //! 对minAreaRect获得的最小外接矩形，用纵横比进行判断
    public boolean verifySizes(RotatedRect mr)
    {
        float error = this.error;
        //Spain car plate size: 52x11 aspect 4,7272
        //China car plate size: 440mm*140mm，aspect 3.142857
        float aspect = this.aspect;
        //Set a min and max area. All other patchs are discarded
        //int min= 1*aspect*1; // minimum area
        //int max= 2000*aspect*2000; // maximum area
        int min= 44*14* verifyMin; // minimum area
        int max= 44*14* verifyMax; // maximum area
        //Get only patchs that match to a respect ratio.
        float rmin= aspect-aspect*error;
        float rmax= aspect+aspect*error;

        int area= (int)(mr.size().height() * mr.size().width());
        float r = mr.size().width() / mr.size().height();
        if(r < 1)   r= mr.size().height() / mr.size().width();
        return area >= min && area <= max  &&  r >= rmin && r <= rmax;
    }

    //! 显示最终生成的车牌图像，便于判断是否成功进行了旋转。
    public Mat showResultMat(Mat src, Size rect_size, Point2f center, int index)
    {
        Mat img_crop = new Mat();
        getRectSubPix(src, rect_size, center, img_crop);

        if(debug)
        {
            String str = "image/tmp/debug_crop_.jpg";
            imwrite(str, img_crop);
        }

        Mat resultResized = new Mat();
        resultResized.create(HEIGHT, WIDTH, TYPE);
        resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
        if(debug)
        {
            String str = "image/tmp/debug_resize_"+ Integer.valueOf(index).toString() + ".jpg";
            imwrite(str, resultResized);
        }
        return resultResized;
    }

    //! 定位车牌图像
//! src 原始图像
//! resultVec 一个Mat的向量，存储所有抓取到的图像
//! 成功返回0，否则返回-1
    public int plateLocate(Mat src, Vector<Mat> resultVec)
    {
        Mat src_blur = new Mat();
        Mat src_gray = new Mat();
        Mat grad = new Mat();

        int scale = SOBEL_SCALE;
        int delta = SOBEL_DELTA;
        int ddepth = SOBEL_DDEPTH;

        if(src.data().isNull())
            return -1;

        //高斯模糊。Size中的数字影响车牌定位的效果。
        GaussianBlur( src, src_blur, new Size(gaussianBlurSize, gaussianBlurSize),0, 0, BORDER_DEFAULT);

        if(debug)
        {
            String str = "res/image/tmp/debug_GaussianBlur.jpg";
            imwrite(str, src_blur);
        }

        /// Convert it to gray
        cvtColor( src_blur, src_gray, CV_RGB2GRAY );

        if(debug)
        {
            String str = "res/image/tmp/debug_gray.jpg";
            imwrite(str, src_gray);
        }

        /// Generate grad_x and grad_y
        Mat grad_x = new Mat();
        Mat grad_y = new Mat();
        Mat abs_grad_x = new Mat();
        Mat abs_grad_y = new Mat();

        /// Gradient X
        //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );

        /// Gradient Y
        //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );

        /// Total Gradient (approximate)
        addWeighted( abs_grad_x, SOBEL_X_WEIGHT, abs_grad_y, SOBEL_Y_WEIGHT, 0, grad );

        //Laplacian( src_gray, grad_x, ddepth, 3, scale, delta, BORDER_DEFAULT );
        //convertScaleAbs( grad_x, grad );


        if(debug)
        {
            String str = "image/tmp/debug_Sobel.jpg";
            imwrite(str, grad);
        }

        Mat img_threshold = new Mat();
        threshold(grad, img_threshold, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);
        //threshold(grad, img_threshold, 75, 255, CV_THRESH_BINARY);

        if(debug)
        {
            String str = "res/image/tmp/debug_threshold.jpg";
            imwrite(str, img_threshold);
        }

        Mat element = getStructuringElement(MORPH_RECT, new Size(morphSizeWidth, morphSizeHeight) );
        morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element);

        if(debug)
        {
            String str = "res/image/tmp/debug_morphology.jpg";
            imwrite(str, img_threshold);
        }

        //Find 轮廓 of possibles plates
        MatVector contours = new MatVector();
        findContours(img_threshold,
                contours, // a vector of contours
                CV_RETR_EXTERNAL, // 提取外部轮廓
                CV_CHAIN_APPROX_NONE); // all pixels of each contours

        Mat result = new Mat();
        if(debug)
        {
            //// Draw blue contours on a white image
            src.copyTo(result);
            drawContours(result,contours,-1,new Scalar(0,0,255,255));
            String str = "res/image/tmp/debug_Contours.jpg";
            imwrite(str, result);
        }

        //Start to iterate to each contour founded

        Vector<RotatedRect> rects = new Vector<RotatedRect>();

        for(int i=0;i<contours.size();++i){
            RotatedRect mr = minAreaRect(contours.get(i));
            if(verifySizes(mr))
                rects.add(mr);
        }


        int k = 1;
        for(int i=0; i< rects.size(); i++)
        {
            RotatedRect minRect = rects.get(i);
            if(verifySizes(minRect))
            {
                // rotated rectangle drawing
                // Get rotation matrix
                // 旋转这部分代码确实可以将某些倾斜的车牌调整正，
                // 但是它也会误将更多正的车牌搞成倾斜！所以综合考虑，还是不使用这段代码。
                // 2014-08-14,由于新到的一批图片中发现有很多车牌是倾斜的，因此决定再次尝试
                // 这段代码。

                /*
                if(debug)
                {
                    Point2f rect_points[] = new Point2f[4];
                    minRect.points(rect_points[0]);
                    for( int j = 0; j < 4; j++ )
                        line( result, rect_points[j], rect_points[(j+1)%4], new Scalar(0,255,255,255), 1, 8 ,0);
                }*/

                float r = minRect.size().width() / minRect.size().height();
                float angle = minRect.angle();
                Size rect_size = new Size((int)minRect.size().width(),(int)minRect.size().height());
                if (r < 1)
                {
                    angle = 90 + angle;
                    //swap(rect_size.width(), rect_size.height());
                    rect_size = new Size(rect_size.height(),rect_size.width());
                }
                //如果抓取的方块旋转超过m_angle角度，则不是车牌，放弃处理
                if (angle - this.angle < 0 && angle + this.angle > 0)
                {
                    //Create and rotate image
                    Mat rotmat = getRotationMatrix2D(minRect.center(), angle, 1);
                    Mat img_rotated = new Mat();
                    //warpAffine(src, img_rotated, rotmat, src.size(), CV_INTER_CUBIC);
                    warpAffine(src,img_rotated,rotmat,src.size());
                    Mat resultMat = showResultMat(img_rotated, rect_size, minRect.center(), k++);
                    resultVec.add(resultMat);
                }
            }
        }
        if(debug)
        {
            String str = "res/image/tmp/debug_result.jpg";
            imwrite(str, result);
        }

        return 0;
    }


    //! 设置与读取变量
    public void setGaussianBlurSize(int gaussianBlurSize){ this.gaussianBlurSize = gaussianBlurSize;}
    public final int getGaussianBlurSize() { return this.gaussianBlurSize;}

    public void setMorphSizeWidth(int morphSizeWidth){ this.morphSizeWidth = morphSizeWidth;}
    public final int getMorphSizeWidth() { return this.morphSizeWidth;}

    public void setMorphSizeHeight(int morphSizeHeight){ this.morphSizeHeight = morphSizeHeight;}
    public final int getMorphSizeHeight() { return this.morphSizeHeight;}

    public void setVerifyError(float error){ this.error = error;}
    public final float getVerifyError() { return this.error;}
    public void setVerifyAspect(float aspect){ this.aspect = aspect;}
    public final float getVerifyAspect() { return this.aspect;}

    public void setVerifyMin(int verifyMin){ this.verifyMin = verifyMin;}
    public void setVerifyMax(int verifyMax){ this.verifyMax = verifyMax;}

    public void setJudgeAngle(int angle){ this.angle = angle;}

    //! 是否开启调试模式
    public void setDebug(boolean debug){ this.debug = debug;}

    //! 获取调试模式状态
    public boolean getDebug(){ return debug;}


    //! PlateLocate所用常量
    public static final int DEFAULT_GAUSSIANBLUR_SIZE = 5;
    public static final int SOBEL_SCALE = 1;
    public static final int SOBEL_DELTA = 0;
    public static final int SOBEL_DDEPTH = CV_16S;
    public static final int SOBEL_X_WEIGHT = 1;
    public static final int SOBEL_Y_WEIGHT = 0 ;
    public static final int DEFAULT_MORPH_SIZE_WIDTH = 17;
    public static final int DEFAULT_MORPH_SIZE_HEIGHT = 3;

    //! showResultMat所用常量
    public static final int WIDTH = 136;
    public static final int HEIGHT = 36;
    public static final int TYPE = CV_8UC3;

    //! verifySize所用常量
    public static final int DEFAULT_VERIFY_MIN = 3;
    public static final int DEFAULT_VERIFY_MAX = 20;

    final float DEFAULT_ERROR = 0.6f;
    final float DEFAULT_ASPECT = 3.75f;
    //! 角度判断所用常量
    public static final int DEFAULT_ANGLE = 30;

    //! 是否开启调试模式常量
    public static final boolean DEFAULT_DEBUG = true;

    //! 高斯模糊所用变量
    protected int gaussianBlurSize;

    //! 连接操作所用变量
    protected int morphSizeWidth;
    protected int morphSizeHeight;

    //! verifySize所用变量
    protected float error;
    protected float aspect;
    protected int verifyMin;
    protected int verifyMax;

    //! 角度判断所用变量
    protected int angle;

    //! 是否开启调试模式，0关闭，非0开启
    protected boolean debug;

}
