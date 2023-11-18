# 测试用的命令，请在afl-2.52b下面进行测试
## 测试之前所需要的必须步骤

利用conda搭建3.9环境(或者有3.9环境的python也可以)
```
$ conda create -n work python=3.9
$ conda activate work
```
安装相关库(注意这里使用的cuda的版本，cuda的版本不一定要统一，但是其他python库的版本必须统一)
```
$ pip install torch==1.12.0+cu113 torchaudio==0.12.0+cu113 torchvision==0.13.0+cu113 pandas tqdm einops timm flask 
```
安装afl-cov

~~~
$ apt-get install lcov
$ unzip afl-cov-master.zip
然后就可以看到一个afl-cov-master的文件夹
~~~

## 安装相关软件

**安装binutils（nm、readelf、objdump要用）**

~~~
$ tar -xf binutils-2.27.tar.gz
$ cd binutils-2.27
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下:
$ ./configure CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make
(如果你想将binutils安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

**安装poppler**

~~~
$ tar -xf poppler-poppler-0.8.tar.gz
$ cd poppler-poppler-0.8
$ apt-get install libfreetype6-dev libfontconfig1-dev libgpm-dev
$ ./autogen.sh
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make
(如果你想将poppler安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

**安装libxml2**

~~~
$ tar -xf libxml2-2.9.2.tar.gz
$ cd libxml2-2.9.2
$ ./autogen.sh
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make
(如果你想将libxml2安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

**安装libpng**

~~~
$ tar -xf libpng-1.6.37.tar.gz
$ cd libpng-1.6.37
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make
(如果你想将libpng安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

**安装libjpeg**

~~~
$ tar -xf jpegsrc.v9e.tar.gz
$ cd jpeg-9e
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage" 
$ make
(如果你想将libjpeg安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

**安装ImageMagick**

~~~
$ tar xvfz ImageMagick-7.1.0-49.tar.gz
$ cd ImageMagick-7.1.0-49
$ sudo apt-get install build-essential libjpeg-dev libpng-dev libtiff-dev libgif-dev zlib1g-dev libfreetype6-dev libfontconfig1-dev
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage" 
$ make
(如果你想将ImageMagick安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

**安装mp3gain**

~~~
$ mkdir mp3gain
$ mv mp3gain-1_5_2-src.zip mp3gain/
$ unzip mp3gain-1_5_2-src.zip
$ vim Makefile
修改其中CCS的值，把gcc改为$afl-gcc$，$afl-gcc$是afl编译器的路径
CC=$afl-gcc$ -fprofile-arcs -ftest-coverage
比如举个例子:
CC=/root/software/afl-2.52b/afl-gcc -fprofile-arcs -ftest-coverage
$ make
(如果你想将mp3gain安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

**安装libpcap和tcpdump**

~~~
$ tar -zxvf libpcap-1.6.2.tar.gz
$ cd libpcap-1.6.2
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage" 
$ make & sudo make install
$ tar -zxvf tcpdump-4.6.2.tar.gz
$ cd tcpdump-4.6.2
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make
(如果你想将tcpdump安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

**安装libtiff**

~~~
$ tar -xf libtiff-Release-v3-9-7.tar.gz
$ cd libtiff-Release-v3-9-7
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make
(如果你想将libtiff安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

**安装zlib**

~~~
$ tar -xf zlib-1.2.11.tar.gz
$ cd zlib-1.2.11
其中$afl-gcc$是这个编译器的路径，这个编译器可以在afl-2.52b文件夹中找到
$ CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" ./configure
比如举个例子，可以写成如下
$ CC="/root/software/afl-2.52b/afl-gcc -fprofile-arcs -ftest-coverage" ./configure
$ make
(如果你想将zlib安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

**安装ncurses**

~~~
$ tar -xf ncurses-6.1.tar.gz
$ cd ncurses-6.1
$ sudo apt-get install libncurses5-dev libncursesw5-dev
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make
(如果你想将ncurses安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

## 测试模型

为了保证模型一开始就是被调用的状态这里我们开启一个服务器来启用模型，之后调用模型只需要发送request请求即可

进入CtoPython/PyDOC下（如果端口被占用了记得修改module_app.py与module_client.py里面的端口信息，默认是http://127.0.0.1:80/）

```
$ python module_app.py
```

**如果你想调用自己的模型，只需要修改module_app.py中的get_output函数即可，模仿着进行修改即可**

进入afl-2.52b文件夹

```
$ cd afl-2.52b
```
### 下面是两种测试方案，请根据需要选择相应的方案

**如果想要原始的afl**

~~~
$ cp afl-fuzz1.c afl-fuzz.c
$ make
~~~

**如果想要控制变异的cycle为1**

```
$ cp afl-fuzz2.c afl-fuzz.c
$ make 
```
**如果想要控制调用模型的次数相同**

```
1. 在CtoPython/docset下设置number.txt为0
$ echo 0 > number.txt
2. 在afl-2.52b下面
$ cp afl-fuzz3.c afl-fuzz.c
$ make 
```
如果测试的时候是保证调用模型的次数相同，那么每次测试的时候需要**设置number.txt为0**
另外如果想要修改控制模型的次数，请使用下面的命令，然后在下图中修改值，默认为1000，大概在代码5120行左右
```
$ vim afl-fuzz3.c
```
![image](https://github.com/CSJianYang/Multilingual-Multimodal-NLP/assets/77664227/00eeca71-d3ed-4ac3-9cf9-489f82cd1864)

### 下面是测试不同程序的

**测试objdump的**

~~~
$ ./afl-fuzz -i testcases/others/elf/ -o ../objdump_out ../binutils-2.27/binutils/objdump -x -a -d @@
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../objdump_out -e "../binutils-2.27/binutils/objdump -x -a -d AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite
$ python afl-showmap.py -f objdump_out -p objdump
记得最后将objdump_out文件夹打包以供数据分析和统计
~~~

**测试readelf的**

~~~
$ ./afl-fuzz -i testcases/others/elf/ -o ../readelf_out ../binutils-2.27/binutils/readelf -a @@
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../readelf_out -e "../binutils-2.27/binutils/readelf -a AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite
$ python afl-showmap.py -f readelf_out -p readelf
记得最后将readelf_out文件夹打包以供数据分析和统计
~~~

**测试nm的**

~~~
$ ./afl-fuzz -i testcases/others/elf/ -o ../nm_out ../binutils-2.27/binutils/nm-new -a @@
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../nm_out -e "../binutils-2.27/binutils/nm-new -a AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite
$ python afl-showmap.py -f nm_out -p nm
记得最后将nm_out文件夹打包以供数据分析和统计
~~~

**测试poppler的**

~~~
$ ./afl-fuzz -i ../pdf_in/ -o ../pdf_out ../poppler-poppler-0.8/utils/pdftotext @@ /dev/null
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../pdf_out -e "../poppler-poppler-0.8/utils/pdftotext AFL_FILE /dev/null" -c ../poppler-poppler-0.8/utils --enable-branch-coverage --overwrite
记得最后将pdf_out文件夹打包以供数据分析和统计
~~~

**测试libxml2的**

~~~
$ ./afl-fuzz -i ../xml_in/ -o ../xml_out ../libxml2-2.9.2/xmllint --valid --recover @@
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../xml_out -e "../libxml2-2.9.2/xmllint --valid --recover AFL_FILE" -c ../libxml2-2.9.2 --enable-branch-coverage --overwrite
$ python afl-showmap.py -f xml_out -p xmllint
记得最后将xml_out文件夹打包以供数据分析和统计
~~~

**测试libpng的**

~~~
$ ./afl-fuzz -i ../png_in/ -o ../png_out ../libpng-1.6.37/pngtest @@ /dev/null
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../png_out -e "../libpng-1.6.37/pngtest AFL_FILE" -c ../libpng-1.6.37 --enable-branch-coverage --overwrite
记得最后将png_out文件夹打包以供数据分析和统计
~~~

**测试libjpeg的**

~~~
$ ./afl-fuzz -i ../jpg_in/ -o ../jpg_out ../jpeg-9e/jpegtran @@
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../jpg_out -e "../jpeg-9e/jpegtran AFL_FILE" -c ../jpeg-9e --enable-branch-coverage --overwrite
$ python afl-showmap.py -f jpg_out -p jpegtran
记得最后将jpg_out文件夹打包以供数据分析和统计
~~~

**测试ImageMagick的**

~~~
$ ./afl-fuzz -i testcases/images/gif -o ../gif_out ../ImageMagick-7.1.0-49/utilities/magick identify @@
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../gif_out -e "../ImageMagick-7.1.0-49/utilities/magick identify AFL_FILE" -c ../ImageMagick-7.1.0-49/utilities --enable-branch-coverage --overwrite
$ python afl-showmap.py -f gif_out -p magick
记得最后将gif_out文件夹打包以供数据分析和统计
~~~

**测试mp3gain的**

~~~
$ ./afl-fuzz -i ../mp3_in/ -o ../mp3_out ../mp3gain/mp3gain @@
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../mp3_out -e "../mp3gain/mp3gain AFL_FILE" -c ../mp3gain --enable-branch-coverage --overwrite
$ python afl-showmap.py -f mp3_out -p mp3gain
记得最后将mp3_out文件夹打包以供数据分析和统计
~~~

**测试tcpdump的**

~~~
$ ./afl-fuzz -i ../pcap_in/ -o ../pcap_out ../tcpdump-4.6.2/tcpdump -e -vv -nr @@
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../pcap_out -e "../tcpdump-4.6.2/tcpdump -e -vv -nr AFL_FILE" -c ../tcpdump-4.6.2 --enable-branch-coverage --overwrite
记得最后将pcap_out文件夹打包以供数据分析和统计
~~~

**测试libtiff的**

~~~
$ ./afl-fuzz -i ../tiff_in/ -o ../tiff_out ../libtiff-Release-v3-9-7/tools/tiffsplit @@
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../tiff_out -e "../libtiff-Release-v3-9-7/tools/tiffsplit AFL_FILE" -c ../libtiff-Release-v3-9-7/tools/ --enable-branch-coverage --overwrite
$ python afl-showmap.py -f tiff_out -p tiffsplit
记得最后将tiff_out文件夹打包以供数据分析和统计
~~~

**测试zlib的**

~~~
$ ./afl-fuzz -i ../zlib_in/ -o ../zlib_out ../zlib-1.2.11/minigzip @@
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../zlib_out -e "../zlib-1.2.11/minigzip AFL_FILE" -c ../zlib-1.2.11 --enable-branch-coverage --overwrite
记得最后将zlib_out文件夹打包以供数据分析和统计
~~~

**测试tic的**

~~~
$ ./afl-fuzz -i testcases/images/text -o ../tic_out ../ncurses-6.1/progs/tic -o /dev/null @@
Fuzz完毕后请先截图最后的运行界面，再进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../tic_out -e "../ncurses-6.1/progs/tic -o /dev/null AFL_FILE" -c ../ncurses-6.1/progs --enable-branch-coverage --overwrite
记得最后将tic_out文件夹打包以供数据分析和统计
~~~

测试的时候如果运行afl-fuzz出现**Pipe at the begining of 'core pattern'**，请按下面步骤进行后再试着运行afl-fuzz

~~~
sudo su
echo core >/proc/sys/kernel/core_pattern
~~~

每次**测试完以后记得直接截图保存以便统计数据**，因为可能会出现乱码，需要重新打开终端才会恢复，暂时没找到解决方法。并且**请将输出文件夹打包保存下来**，用来探索其他指标时使用，比如pcap_out打包成一个zip文件保存下来
![image](https://github.com/CSJianYang/Multilingual-Multimodal-NLP/assets/77664227/f44c2fad-7bee-402d-ab74-818afa68787b)

