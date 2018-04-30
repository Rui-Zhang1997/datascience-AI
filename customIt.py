def getI():
    i = 0
    while True:
        yield i * i
        i += 1

for k in getI():
    print(k)
