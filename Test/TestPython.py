def h():
    print("你好")
    yield 5
    print("再见")


for n in h():
    print(n)


def h():
    print("Wen Chuan")
    m = yield 5  # Fighting!
    print(m)
    d = yield 12
    print('We are together!')


c = h()
c.__next__()  # 相当于c.send(None)
c.send('Fighting!')  # (yield 5)表达式被赋予了'Fighting!'


def h1():
    print('houshuai')
    m = yield 5  # Fighting!
    print(m)
    d = yield 12
    print('We are together!')


c = h1()
m = c.__next__()  # m 获取了yield 5 的参数值 5
d = c.send('Fighting!')  # d 获取了yield 12 的参数值12
print('We will never forget the date', m, '.', d)
