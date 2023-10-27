## 测试用的命令，请在afl-2.52b下面进行测试
### 测试之前所需要的必须步骤

利用conda搭建3.9环境(或者有3.9环境的python也可以)
```
$ conda create -n work python=3.9
$ conda activate work
```
安装相关库(注意这里使用的cuda的版本，cuda的版本不一定要统一，但是其他python库的版本必须统一)
```
$ pip install torch==1.12.0+cu113 torchaudio==0.12.0+cu113 torchvision==0.13.0+cu113 pandas tqdm einops timm
```
进入afl-2.52b文件夹
```
$ cd afl-2.52b
```
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

**测试objdump的**

timeout 12h ./afl-fuzz -i testcases/others/elf/ -o ./objdump ../binutils-2.27/binutils/objdump -x -a -d testcases/others/elf/small_exec.elf @@

**测试readelf的**

timeout 12h ./afl-fuzz -i testcases/others/elf/ -o ./readelf ../binutils-2.27/binutils/readelf -a testcases/others/elf/small_exec.elf @@

**测试nm的**

timeout 12h ./afl-fuzz -i testcases/others/elf/ -o ./nm ../binutils-2.27/binutils/nm-new -a testcases/others/elf/small_exec.elf @@
