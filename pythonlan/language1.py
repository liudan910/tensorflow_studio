list = [1,2,3,4,5];
list1 = ['phy','chemistry',1997]

print("list[0]:", list[0])
print("list1[:3]:" ,list1[:3])

#更新

list1[0] = 'cha'
print("list1:", list1)
del[list1[2]]
print("list1:", list1)

print("------------------------")

print(len([1,2,3]))
print([1,2,3]+[4,5,6])
print(['Hi!']*4)
print( 3 in [1,2,3])
for x in [1,2,3]:
    print(x)

rows = 3
cols = 2
list = [[0 for col in range(cols)] for row in range(rows)]
list[0].append(3)
list[0].append(5)
list[2].append(7)
print(list)
print("------------------------")

#元组：与列表类似 ，区别是不能修改
tup1 = ('physics', 'chemistry', 1997, 2000);
tup2 = (1, 2, 3, 4, 5);
tup3 = "a", "b", "c", "d";
print(tup3)
tup = ()
#元组中只包含一个元素时，需要在元素后面添加逗号

print(tup1[1])
tup4 = tup1 + tup2
print(tup4)

#任意无符号的对象，以逗号隔开，默认为元组
print('abc', -4.24e93, 'xyz')
x, y = 1, 2
print("value of x,y:", x, y)
print("------------------------")

#字典 参考java map . 键唯一
dict = {'Alice': '2341', 'Beth':'9102'} #类型一致
dict2 = {'abc': 123, 98.6: 37}  #类型不一致

print(dict['Beth'])
#print(dict['BB'])

dict2['abc'] = 8
print(dict2['abc'])
print(dict2)
del(dict2[98.6])
print(dict2)

dict = {'Alice': '2341', 'Alice': '3341', 'Beth':'9102'}
print(dict)
#键不可变。所以可以用数字，字符串或元组充当.用列表就不行
#dict = {['Name']: 'Zara', 'Age': 7};
dict = {('Name'): 'Zara', 'Age': 7};
print(dict)

#字典值可以是任意数值类型
dict = {"a": [1, 2]}
print(dict)