package org.easypr.util;

import org.bytedeco.javacpp.BytePointer;

/**
 * There are 3 kinds of convert functions:
 * 1. [float|double|int|long] to[Float|Double|Int|Long](BytePointer pointer)
 * 2. byte[] getBytes([float|double|int|long] value)
 * 3. [float|double|int|long] to[Float|Double|Int|Long](byte[] value)
 * 
 * @author lin.yao
 * 
 */
public class Convert {

    public static float toFloat(BytePointer pointer) {
        byte[] buffer = new byte[4];
        pointer.get(buffer);
        return toFloat(buffer);
    }

    public static double toDouble(BytePointer pointer) {
        byte[] buffer = new byte[8];
        pointer.get(buffer);
        return toDouble(buffer);
    }

    public static int toInt(BytePointer pointer) {
        byte[] buffer = new byte[4];
        pointer.get(buffer);
        return toInt(buffer);
    }

    public static long toLong(BytePointer pointer) {
        byte[] buffer = new byte[8];
        pointer.get(buffer);
        return toLong(buffer);
    }

    public static byte[] getBytes(float value) {
        return getBytes(Float.floatToIntBits(value));
    }

    public static byte[] getBytes(double value) {
        return getBytes(Double.doubleToLongBits(value));
    }

    public static byte[] getBytes(int value) {
        final int length = 4;
        byte[] buffer = new byte[length];
        for (int i = 0; i < length; ++i)
            buffer[i] = (byte) ((value >> (i * 8)) & 0xFF);
        return buffer;
    }

    public static byte[] getBytes(long value) {
        final int length = 8;
        byte[] buffer = new byte[length];
        for (int i = 0; i < length; ++i)
            buffer[i] = (byte) ((value >> (i * 8)) & 0xFF);
        return buffer;
    }

    public static int toInt(byte[] value) {
        final int length = 4;
        int n = 0;
        for (int i = 0; i < length; ++i)
            n += (value[i] & 0xFF) << (i * 8);
        return n;
    }

    public static long toLong(byte[] value) {
        final int length = 8;
        long n = 0;
        for (int i = 0; i < length; ++i)
            n += ((long) (value[i] & 0xFF)) << (i * 8);
        return n;
    }

    public static double toDouble(byte[] value) {
        return Double.longBitsToDouble(toLong(value));
    }

    public static float toFloat(byte[] value) {
        return Float.intBitsToFloat(toInt(value));
    }
}
