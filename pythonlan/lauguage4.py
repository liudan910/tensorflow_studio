import math
from module1.support import print_func
#模块是一个Python文件
print_func("he")

content = dir(math) # dir() 函数一个排好序的字符串列表，内容是一个模块里定义过的名字。
print(content)


#reload 函数
#python中的包就是文件夹，但该文件夹下必须存在 __init__.py 文件, 该文件的内容可以为空。__int__.py用于标识当前文件夹是一个包。

#文件IO
#键盘输入
#str1 = input("输入:")
#print("你输入的是:", str1)
#打开、关闭文件 : buffering = 0  会有寄存 =1 会寄存行，>1 缓存区大小，负值 ，系统 默认
#def open(file, mode='r', buffering=None, encoding=None, errors=None, newline=None, closefd=True): # known special case of open
#mode : r, rb,r+,rb+,w,wb,w+,wb+,a,ab,a+ab+
#r+ 指用于读写。指针在文件开头 ;
#W模式会 覆盖原来文件
#w+ 用于读写。会覆盖原来的文件。
"""
print("----------------------")
fo = open("foo.txt", "w+")       #会覆盖掉原来的文件
fo.write("www.com!\nvery good good! \nhello how are you !\n");
fo.close()
print(fo.mode)
print(fo.closed)
"""

fo2 = open("foo.txt", "r+")
str = fo2.read(10)
print(str)
position = fo2.tell()
print("当前位置" , position)
position = fo2.seek(0,0)
str = fo2.read(10)
print(str)
fo2.close()

#重命名、删除文件
import os
#os.rename("test2.txt","test3.txt")

#os.remove("test2.txt")

#新建,删除目录
#os.mkdir("test")
#os.rmdir("test")
#显示当前的工作目录
print(os.getcwd())

#File 对象方法: file对象提供了操作文件的一系列方法。
#OS 对象方法: 提供了处理文件及目录的一系列方法。
