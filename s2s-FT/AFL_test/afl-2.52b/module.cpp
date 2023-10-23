#include <iostream> 
#include <Python.h>
#include <stdio.h>
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
	char* RunCode(char data[])
	{
    	// 1、初始化python接口
    	Py_Initialize();
    	if (!Py_IsInitialized())
    	{
        	printf("init failed\n");
        	return NULL;
    	}

    	// 2、初始化python系统文件路径，保证可以访问到 .py文件
   	 PyRun_SimpleString("import sys");
   	 PyRun_SimpleString("sys.path.insert(0,'/home/li/CtoPython/script')");

    	// 3、调用python文件名，不用写后缀
    	PyObject* module = PyImport_ImportModule("moduleload");
   	 if (module == nullptr)
    	{
        	printf("module import failed\n");
       		return NULL;
    	}
    	// 4、获取函数对象
    	PyObject* func = PyObject_GetAttrString(module, "run");
    	if (!func || !PyCallable_Check(func))
    	{
        	printf("function not found: say\n");
        	return NULL;
    	}

    	PyObject* args = PyTuple_New(1);
    	// 5：第一个参数，传入 string 类型的值 data
    	PyTuple_SetItem(args, 0, Py_BuildValue("s", data));
    
   	 // 6、调用函数
    	PyObject* ret = PyObject_CallObject(func, args);

    	// 7、接收python计算好的返回值
    	char* result=NULL;
    	// s表示转换成string型变量。
    	// 在这里，最需要注意的是：PyArg_Parse的最后一个参数，必须加上“&”符号
    	PyArg_Parse(ret, "s", &result);
    	std::cout << "result is " << result << std::endl;

    	// 8、结束python接口初始化
    	Py_Finalize();
    	return result;
	}
#ifdef __cplusplus
}
#endif /* __cplusplus */
