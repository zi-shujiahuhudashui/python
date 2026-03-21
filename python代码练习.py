# print("hello,world")
#number=[1,2,4,7,5]
"""
 if 1 in number:
    print("在")

if 9 in number:
    print("在")
else:
    print("不在")
"""
# print(number[3])
# print(max(number))
# number.sort()
# print(number)
# print(list(reversed(number)))
'''
number.append(6)
number.insert(2,3)
print(number)
'''


# print(number.index(7,2,5))
# 元组的解包（拆包）
'''
t = (1, 4, 5)
x, y, z = t
print(x, z, y)
'''

'''
information ={"name":"chenjunming","age": 19}
information["name"] = "shujiahuhudashui"
print(information["name"])
print(information["age"])
# print(information.get("nam"))
# print(information.popitem())
print(information.keys())
print(information.values())
print(information.items())

for key in information:
    print(key,information[key])
for key, value in information.items():
    print(key,value)
if "name" in information:
    print("存在")
'''

# def cifang(a,b):

  #  return a**b
# result=cifang(a=2,b=3)
# print(result)
'''
def greet(name, msg):
    print(f"{msg}, {name}")

greet("Alice", "Hello")  # 正确
# greet("Alice")         # 错误，缺少一个参数
'''

x = 10  # 全局变量
'''
def func():
    global x
    x = 20  # 修改全局变量
    y = 5   # 局部变量
    print(x, y)

func()
print(x)  # 20
'''
'''
add =lambda a,b: a*b
a = int(input("请输入a" ))
b = int(input("请输入b" ))
print(add(a,b))
'''
#模块和包
'''
import math
print(math.sqrt(16))
import test
'''









