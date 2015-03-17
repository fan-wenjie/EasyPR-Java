package org.easypr.util;

import org.bytedeco.javacpp.BytePointer;
import static org.bytedeco.javacpp.opencv_core.*;



/*
 * Created by Anonymous on 2015-03-16.
 */
public class MatHelper {
    public static Object getElement(Mat mat,int... i){
        BytePointer pointer;
        byte []buffer;
        switch (i.length){
            case 0:
                pointer = mat.ptr();
                break;
            case 1:
                pointer = mat.ptr(i[0]);
                break;
            case 2:
                pointer = mat.ptr(i[0],i[1]);
                break;
            default:
                pointer = mat.ptr(i[0],i[1],i[2]);
                break;
        }
        switch (mat.depth()){
            case CV_8U:
            case CV_8S:
                return pointer.get();
            case CV_16U:
            case CV_16S:
                return (short)((pointer.get(0)&0xFF)+((pointer.get(1)&0xFF)<<8));
            case CV_32S:
                buffer = new byte[4];
                pointer.get(buffer);
                return toInt(buffer);
            case CV_32F:
                buffer = new byte[4];
                pointer.get(buffer);
                return toFloat(buffer);
            case CV_64F:
                buffer = new byte[8];
                pointer.get(buffer);
                return toDouble(buffer);
            default:
                return pointer.get();
        }
    }


    public static void setElement(Mat mat,Object value,int... i){
        BytePointer pointer;
        switch (i.length){
            case 0:
                pointer = mat.ptr();
                break;
            case 1:
                pointer = mat.ptr(i[0]);
                break;
            case 2:
                pointer = mat.ptr(i[0],i[1]);
                break;
            default:
                pointer = mat.ptr(i[0],i[1],i[2]);
                break;
        }
        switch (mat.depth()){
            case CV_8U:
            case CV_8S:
                pointer.put((Byte)value);
                break;
            case CV_16U:
            case CV_16S:
                pointer.put(new byte[]{(byte)((Short)value&0xFF),(byte)(((Short)value>>8)&0xFF)});
                break;
            case CV_32S:
                pointer.put(getBytes((Integer) value));
                break;
            case CV_32F:
                pointer.put(getBytes((Float) value));
                break;
            case CV_64F:
                pointer.put(getBytes((Double) value));
                break;
            default:
                break;
        }
    }

    public static byte[] getBytes(float value){
        return getBytes(Float.floatToIntBits(value));
    }

    public static byte[] getBytes(double value){
        return getBytes(Double.doubleToLongBits(value));
    }

    public static byte[] getBytes(int value){
        final int length = 4;
        byte[] buffer = new byte[length];
        for(int i=0;i<length;++i)
            buffer[i] = (byte)((value>>(i*8))&0xFF);
        return buffer;
    }

    public static byte[] getBytes(long value){
        final int length = 8;
        byte[] buffer = new byte[length];
        for(int i=0;i<length;++i)
            buffer[i] = (byte)((value>>(i*8))&0xFF);
        return buffer;
    }

    public static int toInt(byte[] value){
        final int length = 4;
        int n = 0;
        for(int i=0;i<length;++i)
            n += (value[i]&0xFF)<<(i*8);
        return n;
    }

    public static long toLong(byte[] value){
        final int length = 8;
        long n = 0;
        for(int i=0;i<length;++i)
            n += ((long)(value[i]&0xFF))<<(i*8);
        return n;
    }

    public static double toDouble(byte[] value){
        return Double.longBitsToDouble(toLong(value));
    }

    public static float toFloat(byte[] value){
        return Float.intBitsToFloat(toInt(value));
    }
}
