EasyPR-Java
======

Introduction 简介
------------
EasyPR-Java是[liuruoze/EasyPR](https://github.com/liuruoze/EasyPR) 的Java版本。
EasyPR是一个中文的开源车牌识别系统，其目标是成为一个简单、高效、准确的车牌识别引擎。

假设我们有如下的原始图片，需要识别出中间的车牌字符与颜色：

![EasyPR 原始图片](res/image/test_image/plate_recognize.jpg)

经过EasyPR的第一步处理车牌检测（PlateDetect）以后，我们获得了原始图片中仅包含车牌的图块：

![EasyPR 车牌](res/image/test_image/chars_segment.jpg)

接着，我们对图块进行OCR过程，在EasyPR中，叫做字符识别（CharsRecognize）。我们得到了一个包含车牌颜色与字符的字符串：

“蓝牌：苏EUK722”
 
Release Notes 更新
------------
[v0.1 first release](https://github.com/mumu10/EasyPR-Java/releases)
This is the first release can recognize plate in some simple cases.

Downloads and Installation 下载安装
------------
Git克隆一份拷贝到你本机或者直接下载zip压缩。EasyPR-Java 支持以下两种平台：

#### Eclipse
使用Eclipse直接导入EasyPR的目录。

#### INTELLIJ IDEA
本fork版本没有测试

Required Software
------------
本版本在以下平台测试通过：
* windows7 64bit
* Eclipse (Luna)
* jdk1.8.0_45
* junit 4
* javacv 0.11


