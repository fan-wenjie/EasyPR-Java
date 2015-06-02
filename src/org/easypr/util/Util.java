package org.easypr.util;

import java.io.File;
import java.util.Vector;


/**
 * @author lin.yao
 *
 */
public class Util {

    /**
     * get all files under the directory path
     * 
     * @param path
     * @param files
     */
    public static void getFiles(final String path, Vector<String> files) {
        getFiles(new File(path), files);
    }

    /**
     * delete and create a new directory with the same name
     * 
     * @param dir
     */
    public static void recreateDir(final String dir) {
        new File(dir).delete();
        new File(dir).mkdir();
    }
    
    private static void getFiles(final File dir, Vector<String> files) {
        File[] filelist = dir.listFiles();
        for (File file : filelist) {
            if (file.isDirectory()) {
                getFiles(file, files);
            } else {
                files.add(file.getAbsolutePath());
            }
        }
    }
}
