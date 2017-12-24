#异常处理

try:
    #fh = open("testfile.txt","w")
    #fh.write("一个测试文件")
    fh = open("testfile5.txt", "r")
except IOError:
    print("Error:" + "没有找到文件，或读文件失败")
else:        #如果没有异常，则执行else
    print("文件操作成功") #保护不抛出异常的代码
    fh.close()
finally:
    print("finally")
