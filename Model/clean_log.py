adad = open("output_adadelta.log")
lst=[]
for i in adad.readlines():
    if i.startswith('E'):
        lst.append(i.strip())
    elif i.find('val_loss')!=-1:
        lst.append(i.strip())
for i in lst:
    print i
    with open('t.csv','w') as output:
        for i in lst:
            output.write(i)