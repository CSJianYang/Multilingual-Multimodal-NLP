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
## 安装相关软件

**安装poppler**

~~~
$ tar -xf poppler-poppler-0.8.tar.gz
$ cd poppler-poppler-0.8
$ apt-get install libfreetype6-dev libfontconfig1-dev libgpm-dev
$ ./autogen.sh
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC=”$afl-gcc$” CXX=”$afl-g++$”
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/root/software/afl-2.52b/afl-gcc" CXX="/root/software/afl-2.52b/afl-g++"
$ make
(如果你想将poppler安装的话，可以继续执行sudo make install)
~~~

**安装libxml2**

~~~
$ tar -xf libxml2-2.9.2.tar.gz
$ cd libxml2-2.9.2
$ ./autogen.sh
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC=”$afl-gcc$” CXX=”$afl-g++$”
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/root/software/afl-2.52b/afl-gcc" CXX="/root/software/afl-2.52b/afl-g++"
$ make
(如果你想将libxml2安装的话，可以继续执行sudo make install)
~~~

**安装libpng**

~~~
$ tar -xf libpng-1.6.37.tar.gz
$ cd libpng-1.6.37
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC=”$afl-gcc$” CXX=”$afl-g++$”
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/root/software/afl-2.52b/afl-gcc" CXX="/root/software/afl-2.52b/afl-g++" 
$ make
(如果你想将libpng安装的话，可以继续执行sudo make install)
~~~

**安装libjpeg**

~~~
$ tar -xf jpegsrc.v9e.tar.gz
$ cd jpeg-9e
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC=”$afl-gcc$” CXX=”$afl-g++$”
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/root/software/afl-2.52b/afl-gcc" CXX="/root/software/afl-2.52b/afl-g++" 
$ make
(如果你想将libjpeg安装的话，可以继续执行sudo make install)
~~~

**安装ImageMagick**

~~~
$ tar xvfz ImageMagick-7.1.0-49.tar.gz
$ cd ImageMagick-7.1.0-49
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC=”$afl-gcc$” CXX=”$afl-g++$”
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/root/software/afl-2.52b/afl-gcc" CXX="/root/software/afl-2.52b/afl-g++" 
$ make
(如果你想将ImageMagick安装的话，可以继续执行sudo make install)
~~~



## 测试模型

为了保证模型一开始就是被调用的状态这里我们开启一个服务器来启用模型，之后调用模型只需要发送request请求即可

进入CtoPython/PyDOC下（如果端口被占用了记得修改module_app.py与module_client.py里面的端口信息，默认是http://127.0.0.1:80/）

```
$ python module_app.py
```

进入afl-2.52b文件夹

```
$ cd afl-2.52b
```
### 下面是两种测试方案，请根据需要选择相应的方案

如果想要控制变异的cycle为1

```
$ cp afl-fuzz2.c afl-fuzz.c
$ make 
```
如果想要控制调用模型的次数相同
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
./afl-fuzz -i testcases/others/elf/ -o ./objdump_out ../binutils-2.27/binutils/objdump -x -a -d testcases/others/elf/small_exec.elf @@
~~~

**测试readelf的**

~~~
./afl-fuzz -i testcases/others/elf/ -o ./readelf_out ../binutils-2.27/binutils/readelf -a testcases/others/elf/small_exec.elf @@
~~~

**测试nm的**

~~~
./afl-fuzz -i testcases/others/elf/ -o ./nm_out ../binutils-2.27/binutils/nm-new -a testcases/others/elf/small_exec.elf @@
~~~

**测试poppler的**

~~~
./afl-fuzz -i ../pdf_in/ -o ./pdf_out ../poppler-poppler-0.8/utils/pdftotext @@ /dev/null
~~~

**测试libxml2的**

~~~
./afl-fuzz -i ../xml_in/ -o ./xml_out ../libxml2-2.9.2/xmllint --valid --recover @@
~~~

**测试libpng的**

~~~
./afl-fuzz -i ../png_in/ -o /png_out ../libpng-1.6.37/pngtest @@ /dev/null
~~~

**测试libjpeg的**

~~~
./afl-fuzz -i ../jpg_in/ -o /jpg_out ../jpeg-9e/jpegtran @@
~~~

**测试ImageMagick的**

~~~
./afl-fuzz -i testcases/images/gif -o /gif_out ../ImageMagick-7.1.0-49/utilities/magick identify @@
~~~

每次测试完以后记得直接截图保存以便统计数据，因为可能会出现乱码，需要重新打开终端才会恢复，暂时没找到解决方法
![image](https://github.com/CSJianYang/Multilingual-Multimodal-NLP/assets/77664227/f44c2fad-7bee-402d-ab74-818afa68787b)

