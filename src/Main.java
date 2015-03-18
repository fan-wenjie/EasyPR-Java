package org.easypr;


import java.io.*;
import org.easypr.core.PlateLocate;
import org.easypr.test.Test;

public class Main {

    public static void main(String[] args) {
        //java.io.File file = new File("model/ann.xml");
        String path = "res/model/ann.xml";
        System.out.println(new File(path).exists());
    }
}
