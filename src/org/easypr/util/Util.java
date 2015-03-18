package org.easypr.util;

import java.io.File;
import java.util.Vector;

/*
 * Created by fanwenjie
 * @version 1.1
 */

public class Util {

    public static void getFiles(File dir,Vector<String> files){
        File []filelist = dir.listFiles();
        for(File file : filelist){
            if(file.isDirectory())
                getFiles(file,files);
            else
                files.add(file.getAbsolutePath());
        }
    }

    public static void getFiles(String path, Vector<String> files)
    {
        getFiles(new File(path),files);
    }


    //! 通过文件夹名称获取文件名，不包括后缀
    public static String getFileName(final String file)
    {
        String[] strs = file.split("\\.");
        if(strs.length==1)
            return file;
        int extLength = strs[strs.length-1].length() + 1;
        return file.substring(0,file.length()-extLength);
    }

    //删除并创建新的文件夹
    public static void recreateDir(final String dir){
        new File(dir).delete();
        new File(dir).mkdir();
    }
}
