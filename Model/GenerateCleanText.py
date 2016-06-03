import xml.etree.ElementTree
import re
import string
from nltk.corpus import stopwords
#from nltk.stem.porter import *
from stemming.porter2 import stem
#stemmer = PorterStemmer()
def RemoveStopWords(sentence):
    sentence = [x.lower() for x in sentence]
#Stem depening on Porter Algorithm
    sentence = [stem(x) for x in sentence]
    #Remove Stop Words
    filtered_words = [word for word in sentence if word not in stopwords.words('english')]
    #sentence = map(lambda x:x.lower() if x.lower() not in stopwrods else '',sentence)
    newsen = filter(None, filtered_words)
    return ' '.join(filtered_words)
def cleanhtml(raw_html):
  cleanr =re.compile('<.*?>')
  cleantext = re.sub(cleanr,'', raw_html)
  if (type(raw_html) is str):
        cleantext = cleantext.translate(string.maketrans("",""),"~`!@#$%^&*()_+=-]}[{\|?/.>,<;:'")
  cleantext = re.sub('\n',' ', cleantext)
  cleantext = re.sub('\t',' ', cleantext)
  cleantext = re.sub('"','', cleantext)
  cleantext = RemoveStopWords(cleantext.split())
  return cleantext

class Answers:
    def __init__(self,body,score,parentid):
        self.body = body
        self.ParentId=parentid
        self.score=score

class Questions:
    def __init__(self,body,score,ViewCount):
        self.body = body
        self.score=score
        #Answers Count
        self.ansrcnt=0
        self.ViewCount=ViewCount
    def IncreaseAnswerCout(self):
        self.ansrcnt+=1

print 'Fetching Data'
Qs ={}
As={}
e = xml.etree.ElementTree.parse('Posts.xml').getroot()
posts = e.findall('row')
print 'Start Cleaning Data'
t=0
for post in posts:
    t+=1
    if (t%200==0):
        print 'P:',t
    atributes = post.attrib
    if atributes['PostTypeId']=='1':
        Qs[atributes['Id']]=Questions(cleanhtml(atributes['Body']),atributes['Score'],atributes['ViewCount'])
    if atributes['PostTypeId']=='2':
        As[atributes['Id']]=Answers(cleanhtml(atributes['Body']),atributes['Score'],atributes['ParentId'])
print 'Organizing Data'
for i in As.keys():
    Qs[As[i].ParentId].IncreaseAnswerCout()

print 'Saving Data'

with open('All_programmers.csv','w') as output:
    output.write('{},{},{},{},{},{},{},{}\n'.format('id','score','parent','body','q_score','q_ansrcnt','q_ViewCount','q_body'))
    for i in As.keys():
        try:
            pid=As[i].ParentId
            output.write(u"{},{},{},\"{}\",{},{},{},\"{}\"\n".format(i,As[i].score,pid,As[i].body,Qs[pid].score,Qs[pid].ansrcnt,Qs[pid].ViewCount,Qs[pid].body).encode('utf8'))
        except:
            print u"{},{},{},{}\n".format(i,As[i].score,As[i].ParentId,As[i].body)


'''
with open('Answers.csv','w') as output:
    output.write('{},{},{},{}\n'.format('id','score','parent','body'))
    for i in As.keys():
        try:
            output.write(u"{},{},{},\"{}\"\n".format(i,As[i].score,As[i].ParentId,As[i].body).encode('utf8'))
        except:
            print u"{},{},{},{}\n".format(i,As[i].score,As[i].ParentId,As[i].body)

with open('Questions.csv','w') as output:
    output.write(u"{},{},{},{},\"{}\"\n".format('parent','score','ansrcnt','ViewCount','body'))
    for i in Qs.keys():
        try:
            if Qs[i].ansrcnt>0:
                output.write(u"{},{},{},{},\"{}\"\n".format(i,Qs[i].score,Qs[i].ansrcnt,Qs[i].ViewCount,Qs[i].body).encode('utf8'))
        except:
            print u"{},{},{},{}\n".format(i,Qs[i].score,Qs[i].score,Qs[i].body)
'''

