package org.easypr;


import java.io.*;

import static org.bytedeco.javacpp.opencv_core.*;
import org.easypr.core.PlateLocate;
import org.easypr.test.Test;
import org.easypr.util.Convert;

public class Main {

    public static void main(String[] args) {
        Test.plateDetect();
    }
}
