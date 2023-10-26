## 测试用的命令，请在afl-2.52b下面进行测试
### 测试之前所需要的必须步骤
```
# 利用conda搭建3.9环境(或者有3.9环境的python也可以)
$ conda create -n work python=3.9
# 安装相关库
$ pip install torch==1.12.0+cu113 torchaudio==0.12.0+cu113 torchvision==0.13.0+cu113 pandas tqdm einops
# 进入afl-2.52b文件夹
cd afl-2.52b
# 找到Makefile，对Makefile文件进行修改，在下面这个图这里将-I，-L中改为自己电脑下的/XX/include/python3.9和/XX/lib
vim Makefile
![image](https://github.com/CSJianYang/Multilingual-Multimodal-NLP/assets/77664227/a05ec713-fa71-4938-9e68-5c997a8cb365)
# 如果出现libpython3.9m.so.1.0: cannot open shared object file，请将/XX/lib下的libpython3.9m.so.1.0放入/usr/lib里面
# 对工具进行初始化
make
sudo make install

```
**测试objdump的**

timeout 12h ./afl-fuzz -i testcases/others/elf/ -o ./objdump ../binutils-2.27/binutils/objdump -x -a -d testcases/others/elf/small_exec.elf @@

**测试readelf的**

timeout 12h ./afl-fuzz -i testcases/others/elf/ -o ./readelf ../binutils-2.27/binutils/readelf -a testcases/others/elf/small_exec.elf @@

**测试nm的**

timeout 12h ./afl-fuzz -i testcases/others/elf/ -o ./nm ../binutils-2.27/binutils/nm-new -a testcases/others/elf/small_exec.elf @@

## 测试所需要注意的点
如下图，主要需要在afl-2.52b中的afl-fuzz.c中找到如下位置，大概在5120行左右，首先需要自己创建一个in.txt和一个out.txt，位置可以自己定，但记得修改下图中in.txt和out.txt的位置，然后就是下图中的system调用的python文件，这里用的python文件moduleload-LL.py就是我们运行的模型，可以直接更换。
![图片](./readme的图片.png)

**对于moduleload-LL需要解释的一些东西**：在这个文件中，首先会从创建的in.txt中读取用16进制表示的字符串，需要保证这个字符串的长度是2的倍数，然后处理得到模型的输入，然后运行模型得到输出，通过top_n策略，得到相应的输出\[op,pos,op,pos,op,pos,...\]，我将输出处理成了16进制表示的字符串（每3个16进制表示一个数字），最后会加上00e001方便我们在afl-fuzz.c里面改的源码能够处理输出。
其实改的时候就可以针对于moduleload-LL.py中的run函数中运行模型那一块进行替换即可.
![图片](./readme的图片2.png)
在moduleload-LL.py中同样需要新建err.txt和fragment.txt，位置自定义，而in.txt和out.txt的路径和之前建的一样。

在做完这一切后在afl-2.52b下面运行下面命令更新afl-fuzz


~~~
make
make install
~~~

做完测试所需要做的就可以直接运行最上面的代码分别测试3个程序
