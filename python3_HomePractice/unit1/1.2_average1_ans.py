num = []

def simple_conut(x):
    i = 0
    for _ in x:
        i = i + 1
    return i

def simple_sum(x):
    n = 0
    sum = 0
    for n in x:
        sum = sum + int(n)
    return sum

while True:
    test = input("enter a number or Enter to finish: ")
    if test =="":
        break
    num.append(test)

print("numbers: [",end='')
for x in num:
    print(x,end='')
    if x == num[len(num)-1]:
        break
    print(',',end='')
print(']')
num.sort()
print('count = '+str(simple_conut(num)),end=' ')
print('sum = '+str(simple_sum(num)),end=' ')
print('lowest = '+str(num[0]),end=' ')
print('highest = '+str(num[len(num)-1]),end=' ')
print('mean = '+str(simple_sum(num)/len(num)),end=' ')

      