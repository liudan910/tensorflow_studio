#类
#类变量
#实例变量
#方法重写
#实例变量
#继承
#实例化
#方法
#对象

class Emplyee:
    '所有员工的基类'
    empCount = 0      #类变量，类的所有实例之间共享

    def __init__(self,name,salary): #构造函数，self代表类的实例，self 在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。
        self.name = name
        self.salary = salary
        Emplyee.empCount += 1
    def displayCount(self):
        print("Total Employee %d" % Emplyee.empCount)

    def displayEmployee(self):
        print("name:",self.name,",salary:",self.salary)

    def prt(employee): ##self 不是 python 关键字，我们把他换成 employee 也是可以正常执行的:
        print(employee)
        print(employee.__class__)

emplyee = Emplyee("xx",50 )
emplyee.displayCount()
emplyee.displayEmployee()
emplyee.prt()

#python对象销毁(垃圾回收)
#Python 使用了引用计数这一简单技术来跟踪和回收垃圾。
#析构函数 __del__

#类继承
class Parent:  # 定义父类
    parentAttr = 100

    def __init__(self):
        print( "调用父类构造函数")

    def parentMethod(self):
        print('调用父类方法')

    def setAttr(self, attr):
        Parent.parentAttr = attr

    def getAttr(self):
        print(  "父类属性 :", Parent.parentAttr)
class Child(Parent):
    def __init__(self):
        print("调用子类构造方法")

    def childMethod(self):
        print(  '调用子类方法')
c = Child()
c.childMethod()
c.parentMethod()
c.setAttr(200)
c.getAttr()

#方法重写
#基础重载方法：__int__, __del__,__repr__,  __str__(相当于toString),__cmp__ (compare方法）

#运算符重载

#类的属性与方法

class JustCounter:
    __secretCount = 0 #私有变量 __private_method：两个下划线开头，声明该方法为私有方法，不能在类地外部调用。在类的内部调用
    publicCount = 0 #仅有变量

    def count(self):
        self.__secretCount +=1
        self.publicCount +=1
        print(self.__secretCount)

counter = JustCounter()
counter.count()
counter.count()

print( counter.publicCount)
#print(counter.__secretCount)Python不允许实例化的类访问私有数据，但你可以使用 object._className__attrName 访问属性
print(counter._JustCounter__secretCount)

#单下划线、双下划线、头尾双下划线说明：
#__foo__: 定义的是特殊方法，一般是系统定义名字 ，类似 __init__() 之类的。
#foo: 以单下划线开头的表示的是 protected 类型的变量，即保护类型只能允许其本身与子类进行访问，不能用于 from module import *
#__foo: 双下划线的表示的是私有类型(private)的变量, 只能是允许这个类本身进行访问了。

import json
data = [{'a': 1, 'b': 2, 'c': 3, 'd': 4}]
jsonData = json.dumps(data)
print(data)
print(jsonData)
print(json.dumps({'a':'Runoob','b':7},sort_keys=True, indent=4, separators=(',',":")))

