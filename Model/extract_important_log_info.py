names = ['adadelta','adagrad','adam','SGD','adamax','rmsprop']
for name in names:
    adad = open("output_{}.log".format(name))
    lst=[]
    txt = adad.readline()
    txt= txt.split('\r')
    for i in range(0,len(txt)-1,2):
        epoch =  txt[i][6:txt[i].index('/')]
        tl,vl =  txt[i+1][57:].split(' - ')
        vl = vl[10:]
        #print epoch,tl,vl
        lst.append((epoch,tl,vl))

    with open(name+'.csv','w') as output:
        output.write('\"epoch\",\"train_loss\",\"validation_loss\"\n')
        for i in lst:
            output.write('{},{},{}{}'.format(i[0],i[1],i[2],'\n'))
    print name+" Done"
