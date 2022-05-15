import csv
import string
from farasa.stemmer import FarasaStemmer
from nltk.corpus import stopwords
import unicodedata as ud

stemmer = FarasaStemmer()
file_csv="stopwords.csv"
file_csv2="noStopwords.csv"
data=[]
data2=[]

def getData(categ):
    digits=10
    zeros="000"
    coo=categ
    categ='/'+categ+'/'
    for file_num in range(3000):
        while True:
            if file_num<digits:
                filename=zeros+str(file_num)+".txt"
                break
            digits=digits*10
            zeros=zeros[:len(zeros)-1]
        print(filename)
        with open('./data2'+categ+filename, encoding="utf8") as f:
            lines = f.readlines()
            lines="".join(lines)
            lines = ''.join(c for c in lines if not ud.category(c).startswith('P') and not ud.category(c).startswith('M'))
            lines= stemmer.stem(lines)
            print(lines)
            print("*******************")
            lines2=""
            data.append({"Content":lines,"Category":coo})
            # lines=lines.translate(str.maketrans('','', "".join(stopwords.words("arabic"))))
            #lines="".join([word for word in lines.split(" ") if word not in stopwords.words('english')])
            # lines=''.join(c if c not in "".join(stopwords.words("arabic")))
            for s_word in stopwords.words('arabic'):
                    lines=lines.replace(' '+s_word+' ',' ')
                    lines=lines.replace(' '+s_word+'\n',' ')
                    lines=lines.replace('\n'+s_word+'\n',' ')
                   # print(s_word)
            print( lines)#stopwords.words('arabic'))
            print("*******************")
            data2.append({"Content":lines,"Category":coo})

getData("Culture")
getData("Finance")
getData("Medical")
getData("Politics")
getData("Religion")
getData("Sports")
getData("Technology")

print(data)
with open(file_csv, 'w', encoding='UTF8') as f:
    writer = csv.DictWriter(f, fieldnames=["Content","Category"])
    writer.writeheader()
    writer.writerows(data)
with open(file_csv2, 'w', encoding='UTF8') as f:
    writer = csv.DictWriter(f, fieldnames=["Content","Category"])
    writer.writeheader()
    writer.writerows(data2)



