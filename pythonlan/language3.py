def printme(str):
    print(str)
    return

printme("wq")

#参数传递
#不可更改对象：string,tuples,number
#可更改对象： list,dict

def ChangeInt(a):
    a = 10
b = 2
ChangeInt(b)
print(b)
def changeme( mylist ):
    mylist.append([1,2,3,4]);
    print("函数内值:", mylist)
mylist = [10,20,30]
changeme(mylist)
print("函数外值：", mylist )


#关键字参数 ： 允许调用时与声明时顺序不一致

def printinfo( name, age=35):
    print("name:", name)
    print("age:", age)

printinfo(age=50, name="miki")
printinfo( name="miki")

#不定长参数
print("--------------------")
def printinfo2(arg1,*vartuple):
  #  print(arg1)
    for var in vartuple:
        print(var)
    return

printinfo2( 10 )
printinfo2(70, 60, 50)
print("--------------------")
#匿名函数： lambda表达式
# lambda [arg1 [,arg2,....argn]]:expression
# lambda 参数，参数 ……： 表达式
sum = lambda  arg1,arg2 : arg1+arg2
print("相加：", sum(10,20))


print("--------------------")

#全局变量作用于函数内，需要加global

globvar = 0
def set_globvar_to_one():
    global globvar
    globvar = 1

def print_globvar():
    print(globvar)

set_globvar_to_one()
print(globvar)
print(print_globvar())
