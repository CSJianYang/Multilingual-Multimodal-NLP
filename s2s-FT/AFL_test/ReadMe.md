## 测试用的命令，请在afl-2.52b下面进行测试

**测试objdump的**

timeout 12h ./afl-fuzz -i testcases/others/elf/ -o ./objdump ../binutils-2.27/binutils/objdump -x -a -d testcases/others/elf/small_exec.elf @@

**测试readelf的**

timeout 12h ./afl-fuzz -i testcases/others/elf/ -o ./readelf ../binutils-2.27/binutils/readelf -a testcases/others/elf/small_exec.elf @@

**测试nm的**

timeout 12h ./afl-fuzz -i testcases/others/elf/ -o ./nm ../binutils-2.27/binutils/nm-new -a testcases/others/elf/small_exec.elf @@

## 测试所需要注意的点
如下图，主要需要在afl-2.52b中的afl-fuzz.c中找到如下位置，大概在5120行左右，首先需要自己创建一个in.txt和一个out.txt，位置可以自己定，但记得修改下图中in.txt和out.txt的位置，然后就是下图中的system调用的python文件，这里用的python文件就是我们运行的模型，可以直接更换。
![图片](./readme的图片.png)
