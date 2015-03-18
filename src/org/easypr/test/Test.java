package org.easypr.test;


/*
 * Created by fanwenjie
 * @version 1.1
 */

import java.io.File;

public class Test {
    public String getPath(){
        java.io.File file = new File("model/ann.xml");
        String path = new Object().getClass().getResource("/model/ann.xml").getPath();
        //String path = file.getAbsolutePath();
        return path;
    }
}
