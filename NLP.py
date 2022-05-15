# coding=utf8
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from PIL import Image
# import Django
# import Eel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from farasa.pos import FarasaPOSTagger
from farasa.ner import FarasaNamedEntityRecognizer
from farasa.diacratizer import FarasaDiacritizer
from farasa.segmenter import FarasaSegmenter
from farasa.stemmer import FarasaStemmer
from nltk.stem import SnowballStemmer
from nltk import ISRIStemmer
from tashaphyne import stemming as tsh
from tashaphyne import normalize as norm
import unicodedata as ud
# img_viewer.py
import pickle
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os.path
#sg.theme('DarkTeal12')
import re
from nltk import ISRIStemmer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import threading
import requests
sg.ChangeLookAndFeel('DarkTeal10')
CLEANR = re.compile('<.*?>')## to remove tags
def normalize(text):
    text = norm.normalize_searchtext(text)
    return text
def isrisStemming(text):
  isris=ISRIStemmer()
  arr=text.split(" ")
  arr1=[]
  for a in range(len(arr)):
    arr1.append(isris.stem(arr[a]))
  return " ".join(arr1)
def assemStemming(text):
  assem=SnowballStemmer("arabic")
  arr=text.split(" ")
  arr1=[]
  for a in range(len(arr)):
    arr1.append(assem.stem(arr[a]))
  return " ".join(arr1)
def tashStemming(text):
  tashaphyne=tsh.ArabicLightStemmer()
  arr=text.split(" ")
  arr1=[]
  for a in range(len(arr)):
    arr1.append(tashaphyne.light_stem(arr[a]))
  return " ".join(arr1)
def farasaStemming(text):
    farasa = FarasaStemmer()
    return farasa.stem(text)


#data["Content"]=data['Content'].apply(lambda x: stemming(x) )

def diacritics(text):
    return ''.join(c for c in text if not ud.category(c).startswith('M'))

def cleaning(text):
  ###We need to Convert number words to numeric form first
  return re.sub('([@A-Za-z0-9_ـــــــــــــ]+)|[^\w\s]|#|http\S+', '', text ).strip() # cleaning up
  ###We still need to remove extra white spaces####
def non_Arabic(text):
  return re.sub('([@A-Za-z])', '', text ).strip() # cleaning up
def remove_numbers(text):
  return re.sub('([0-9])', '', text ).strip() # cleaning up
def remove_url(text):
  return re.sub('http\S+', '', text ).strip() # cleaning up
def remove_punc(text):
  return re.sub('[^\w\s]|#', '', text ).strip() # cleaning up
def stopword_remove(text,norm):
  stopwords=u"، ء ءَ آ آب آذار آض آل آمينَ آناء آنفا آه آهاً آهٍ آهِ أ أبدا أبريل أبو أبٌ أجل أجمع أحد أخبر أخذ أخو أخٌ أربع أربعاء أربعة أربعمئة أربعمائة أرى أسكن أصبح أصلا أضحى أطعم أعطى أعلم أغسطس أفريل أفعل به أفٍّ أقبل أكتوبر أل ألا ألف ألفى أم أما أمام أمامك أمامكَ أمد أمس أمسى أمّا أن أنا أنبأ أنت أنتم أنتما أنتن أنتِ أنشأ أنه أنًّ أنّى أهلا أو أوت أوشك أول أولئك أولاء أولالك أوّهْ أى أي أيا أيار أيضا أيلول أين أيّ أيّان أُفٍّ ؤ إحدى إذ إذا إذاً إذما إذن إزاء إلى إلي إليكم إليكما إليكنّ إليكَ إلَيْكَ إلّا إمّا إن إنَّ إى إياك إياكم إياكما إياكن إيانا إياه إياها إياهم إياهما إياهن إياي إيهٍ ئ ا ا? ا?ى االا االتى ابتدأ ابين اتخذ اثر اثنا اثنان اثني اثنين اجل احد اخرى اخلولق اذا اربعة اربعون اربعين ارتدّ استحال اصبح اضحى اطار اعادة اعلنت اف اكثر اكد الآن الألاء الألى الا الاخيرة الان الاول الاولى التى التي الثاني الثانية الحالي الذاتي الذى الذي الذين السابق الف اللاتي اللتان اللتيا اللتين اللذان اللذين اللواتي الماضي المقبل الوقت الى الي اليه اليها اليوم اما امام امس امسى ان انبرى انقلب انه انها او اول اي ايار ايام ايضا ب بؤسا بإن بئس باء بات باسم بان بخٍ بد بدلا برس بسبب بسّ بشكل بضع بطآن بعد بعدا بعض بغتة بل بلى بن به بها بهذا بيد بين بَسْ بَلْهَ ة ت تاء تارة تاسع تانِ تانِك تبدّل تجاه تحت تحوّل تخذ ترك تسع تسعة تسعمئة تسعمائة تسعون تسعين تشرين تعسا تعلَّم تفعلان تفعلون تفعلين تكون تلقاء تلك تم تموز تينك تَيْنِ تِه تِي ث ثاء ثالث ثامن ثان ثاني ثلاث ثلاثاء ثلاثة ثلاثمئة ثلاثمائة ثلاثون ثلاثين ثم ثمان ثمانمئة ثمانون ثماني ثمانية ثمانين ثمنمئة ثمَّ ثمّ ثمّة ج جانفي جدا جعل جلل جمعة جميع جنيه جوان جويلية جير جيم ح حاء حادي حار حاشا حاليا حاي حبذا حبيب حتى حجا حدَث حرى حزيران حسب حقا حمدا حمو حمٌ حوالى حول حيث حيثما حين حيَّ حَذارِ خ خاء خاصة خال خامس خبَّر خلا خلافا خلال خلف خمس خمسة خمسمئة خمسمائة خمسون خمسين خميس د دال درهم درى دواليك دولار دون دونك ديسمبر دينار ذ ذا ذات ذاك ذال ذانك ذانِ ذلك ذهب ذو ذيت ذينك ذَيْنِ ذِه ذِي ر رأى راء رابع راح رجع رزق رويدك ريال ريث رُبَّ ز زاي زعم زود زيارة س ساء سابع سادس سبت سبتمبر سبحان سبع سبعة سبعمئة سبعمائة سبعون سبعين ست ستة ستكون ستمئة ستمائة ستون ستين سحقا سرا سرعان سقى سمعا سنة سنتيم سنوات سوف سوى سين ش شباط شبه شتانَ شخصا شرع شمال شيكل شين شَتَّانَ ص صاد صار صباح صبر صبرا صدقا صراحة صفر صهٍ صهْ ض ضاد ضحوة ضد ضمن ط طاء طاق طالما طرا طفق طَق ظ ظاء ظل ظلّ ظنَّ ع عاد عاشر عام عاما عامة عجبا عدا عدة عدد عدم عدَّ عسى عشر عشرة عشرون عشرين عل علق علم على علي عليك عليه عليها علًّ عن عند عندما عنه عنها عوض عيانا عين عَدَسْ غ غادر غالبا غدا غداة غير غين ـ ف فإن فاء فان فانه فبراير فرادى فضلا فقد فقط فكان فلان فلس فهو فو فوق فى في فيفري فيه فيها ق قاطبة قاف قال قام قبل قد قرش قطّ قلما قوة ك كأن كأنّ كأيّ كأيّن كاد كاف كان كانت كانون كثيرا كذا كذلك كرب كسا كل كلتا كلم كلَّا كلّما كم كما كن كى كيت كيف كيفما كِخ ل لأن لا لا سيما لات لازال لاسيما لام لايزال لبيك لدن لدى لدي لذلك لعل لعلَّ لعمر لقاء لكن لكنه لكنَّ للامم لم لما لمّا لن له لها لهذا لهم لو لوكالة لولا لوما ليت ليرة ليس ليسب م مئة مئتان ما ما أفعله ما انفك ما برح مائة ماانفك مابرح مادام ماذا مارس مازال مافتئ ماي مايزال مايو متى مثل مذ مرّة مساء مع معاذ معه معها مقابل مكانكم مكانكما مكانكنّ مكانَك مليار مليم مليون مما من منذ منه منها مه مهما ميم ن نا نبَّا نحن نحو نعم نفس نفسه نهاية نوفمبر نون نيسان نيف نَخْ نَّ ه هؤلاء ها هاء هاكَ هبّ هذا هذه هل هللة هلم هلّا هم هما همزة هن هنا هناك هنالك هو هي هيا هيهات هيّا هَؤلاء هَاتانِ هَاتَيْنِ هَاتِه هَاتِي هَجْ هَذا هَذانِ هَذَيْنِ هَذِه هَذِي هَيْهات و و6 وأبو وأن وا واحد واضاف واضافت واكد والتي والذي وان واهاً واو واوضح وبين وثي وجد وراءَك ورد وعلى وفي وقال وقالت وقد وقف وكان وكانت ولا ولايزال ولكن ولم وله وليس ومع ومن وهب وهذا وهو وهي وَيْ وُشْكَانَ ى ي ياء يفعلان يفعلون يكون يلي يمكن يمين ين يناير يوان يورو يوليو يوم يونيو ّأيّان"
  #stopwords=normalize(stopwords)
  if norm:
      stopwords=normalize(stopwords)
  stopwords=stopwords.split(" ")
  stopwords.append("يا")
  text=text.split(" ")
  #stopwords=stopwords.apply(lambda x: normalize(x) )
  # print(stopwords)
  return " ".join([w for w in text if not w in stopwords])
#
def learn(savePath,arabic,url,number,punct,stopw,dia,norm,stem,Class,split ):
    if split=="":
        split=33
    if dataFile=="":
      open_window("Choose a .csv file first","ERROR")
      return
    df = pd.read_csv(dataFile, sep=',')
    df=df.dropna(how='any', axis=0)
    print(df.isnull().sum())
    print(df["Category"].value_counts())
    df = df[["Content", "Category"]]
    x = np.array(df["Content"])
    if arabic:
        df["Content"]=df['Content'].apply(lambda x: non_Arabic(x) )
    print(df["Content"][9526])
    if url:
        df["Content"]=df['Content'].apply(lambda x: remove_url(x) )
    if number:
        df["Content"]=df['Content'].apply(lambda x: remove_numbers(x) )
    if punct:
        df["Content"]=df['Content'].apply(lambda x: remove_punc(x) )
    if dia:
        df["Content"]=df['Content'].apply(lambda x: diacritics(x) )
    if norm:
        df["Content"]=df['Content'].apply(lambda x: normalize(x) )
    print(df["Content"][9526])
    if stopw:
        df["Content"]=df['Content'].apply(lambda x: stopword_remove(x,norm) )
    print(df["Content"][9526])
    if stem=="Assem":
        df["Content"]=df['Content'].apply(lambda x: assemStemming(x) )
    elif stem=="ISRIS":
        df["Content"]=df['Content'].apply(lambda x: isrisStemming(x) )
    elif stem=="Tashaphyne":
        df["Content"]=df['Content'].apply(lambda x: tashStemming(x) )
    elif stem=="Farasa":
        df["Content"]=df['Content'].apply(lambda x: farasaStemming(x) )
    df["Content"]=df['Content'].apply(lambda x: re.sub(' +', ' ', x) )
    print(df["Content"][9526])
    x = df["Content"].values.tolist()
    y = df["Category"].values.tolist()
    cv =  TfidfVectorizer()
    X = cv.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(split)/100, random_state=42)
    #"Naive Bayes","Neural Network","SVM","Logistic Regression","Random Forest"
    if Class=="Naive Bayes":
        model = MultinomialNB()
        print("                         Naive Bayes                               ")
    elif Class=="SVM":
        model=svm.LinearSVC()
        print("                         SVM                               ")
    elif Class=="Logistic Regression":
        model=LogisticRegression()
        print("                         Logistic Regression                               ")
    elif Class=="Random Forest":
        model=RandomForestClassifier()
        print("                         Random Forest                               ")
    model.fit(X_train,y_train)
    predicted_out=model.predict(X_test)
    print("accuracy_score     f1_score           precision_score   recall_score")
    print(accuracy_score(y_test, predicted_out),f1_score(y_test, predicted_out,average='macro'),precision_score(y_test, predicted_out,average='macro'), recall_score(y_test, predicted_out,average='macro'))
    cm = confusion_matrix(y_test, predicted_out, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    disp.plot()
    plt.savefig(savePath+'.png')
    with open(savePath+'.txt', 'w') as f:
        preproc=str(arabic)+"#"+str(url)+"#"+str(number)+"#"+str(punct)+"#"+str(stopw)+"#"+str(dia)+"#"+str(norm)+"#"+stem+ "#"+Class+"#"
        preproc+=str(accuracy_score(y_test, predicted_out))+"#"+str(f1_score(y_test, predicted_out,average='macro'))+"#"+str(precision_score(y_test, predicted_out,average='macro'))+"#"+str(recall_score(y_test, predicted_out,average='macro'))
        f.write(preproc)
    pickle.dump(model, open(savePath+".sav", 'wb'))
    pickle.dump(cv, open(savePath+".pickle", "wb")) #Save vectorizer


# First the window layout in 2 columns
def open_window(text,title,s=(350,40)):
    layout = [[sg.Text(text, key="new")]]
    window = sg.Window(title, layout,size=s)
    choice = None
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

    window.close()

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg
def confucious(fileName):
    plot=fileName.replace(".sav",".png")
    im = Image.open(plot)
    # This method will show image in any image viewer
    im.show()

def classifyText(fileName,text):
    load=pickle.load(open(fileName, 'rb'))
    fileName=fileName.replace(".sav",".pickle")
    preproc=fileName.replace(".pickle",'.txt')
    plot=fileName.replace(".pickle",".png")
    #lines=""
    with open(preproc) as f:
        lines = f.readlines()
    #str(arabic)+"#"+str(url)+"#"+str(number)+"#"+str(punct)+"#"+str(stopw)+"#"+str(dia)+"#"+str(norm)+"#"+stem+"#"+Class+ "#"+split
    lines=lines[0].split("#")
    if lines[0]=="True":
        text=non_Arabic(text)
    if lines[1]=="True":
        text=remove_url(text)
    if lines[2]=="True":
        text=remove_numbers(text)
    if lines[3]=="True":
        text=remove_punc(text)
    if lines[5]=="True":
        text=diacritics(text)
    if lines[4]=="True":
        text=stopword_remove(text,lines[5])
    if lines[7]=="Assem":
        text=assemStemming(text)
    elif lines[7]=="ISRIS":
        text=isrisStemming(text)
    elif lines[7]=="Tashaphyne":
        text=tashStemming(text)
    elif lines[7]=="Farasa":
        text=assemStemming(text)
    text=re.sub(' +', ' ', text)
    loadcv=pickle.load( open(fileName, "rb")) #Save vectorizer

    print(text)
    df = loadcv.transform([text]).toarray()
    if lines[8]!="SVM":
        output = "\t\t\t\t\tThis article is about: \n\t\t\tCulture,Finance,Medical,Politics,Religion,Sports,Technology\n\t\t\t"+np.array2string(load.predict_proba(df), precision=2, separator=',    ',
                      suppress_small=True)+"\n"
    else:
        output = "\t\t\t\tThis article is about: "+"".join(load.predict(df))+"\n"
    output+="\t\t\t\t\t"+lines[8]+"\n"
    output+="Accuracy: "+lines[9]+" F1-Score: "+lines[10]+" Precision "+lines[11]+" Recall: "+lines[12]

    return output


column2 = [
    [sg.In(size=(25, 1), enable_events=True, key="file",visible=False),sg.FileBrowse(button_text="Load Model", enable_events=True,key="Load")],
    [sg.Text("No Valid Model Loaded Yet", text_color="Red",enable_events=True, key="noValid")],
    [
        sg.Text("Article Link"),
    ],
    [
        sg.In(size=(25, 1), enable_events=True, key="linkField")],
    [
              sg.Text("Or Article Text"),
    ],
    [
        #sg.Text("Article Text"),
        sg.Multiline(
             enable_events=True, size=(40, 30), key="inputT"
        ),


    ],
    [
        sg.Button(button_text="Classify Link", enable_events=True,key="link"),
        sg.Button(button_text="Classify Text", enable_events=True,key="Text"),
        sg.Button(button_text="Confusion Matrix", enable_events=True,key="Plot")
     ],
]

# For now will only show the name of the file that was chosen
column1 = [
    [sg.Text("Train Your Data")],
    [sg.Text("Choose your dataset (.csv file), please.")],
    [sg.In(size=(25, 1), enable_events=True, key="data"),sg.FileBrowse()],
    [sg.Text("Cleaning the Data:")],
    [sg.Checkbox('Remove Non-Arabic Words', default=False,enable_events=True, key="arabic")],
    [sg.Checkbox('Remove URLs ', default=False,enable_events=True, key="url")],
    [sg.Checkbox('Remove Numbers', default=False,enable_events=True, key="number")],
    [sg.Checkbox('Remove Punctuation and Symbols', default=False,enable_events=True, key="punct")],
    [sg.Checkbox('Remove Stopwords', default=False,enable_events=True, key="stopw")],
    [sg.Checkbox('Remove Diacritics', default=False,enable_events=True, key="dia")],
    [sg.Checkbox('Normalize', default=False,enable_events=True, key="norm")],
    #[sg.Checkbox('My Checkbox', default=False)],
    [sg.Text("Stemmer")],
    [sg.DropDown(enable_events=True,key="Stem",values=["Assem","ISRIS","Tashaphyne","Farasa","No stemming"],default_value="Assem")],
    [sg.Text("Percentage of data used for testing")],
    [sg.In(size=(2, 1), enable_events=True, key="split")],
    [sg.Text("Classifier")],
    [sg.DropDown(enable_events=True,key="Class",values=["Naive Bayes","SVM","Logistic Regression","Random Forest"],default_value="Naive Bayes")],
    [sg.In(size=(1, 1),visible=False, enable_events=True, key="save"),sg.FileSaveAs(button_text="Save Model")],
    [sg.Text("",key="saving")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(column1),
        sg.VSeperator(),
        sg.Column(column2),
        #sg.Column(column3),
        #sg.Column(column4),

    ]
]
window = sg.Window("News Classification", layout)
textInput=""
stemmer="Assem"
path=""
dataFile=""
# Run the Event Loop
while True:
    event, values = window.read(timeout=500)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "link":
        link=values["linkField"]
        if link=='':
            continue
        response = requests.get(link)
        # print (response.status_code)
        res=""
        if path!="":
            res=classifyText(path,re.sub(CLEANR, '', response.text))
            open_window(res,"This Article Is About:",(700,100))
    if event =="Plot":
        if path!="":
            confucious(path)
    # Folder name was filled in, make a list of files in the folder
    if event == "Text":
        #classifier = values["-FOLDER-"]
        res=""
        if path!="":
            res=classifyText(path,textInput)
            open_window(res,"This Article Is About:",(700,100))
        #window["new"].update(res)


    elif event == "inputT":  # A file was chosen from the listbox
        textInput=values["inputT"]
    elif event == "Class":
        classifier=values["Class"]
    elif event == "data":
        dataFile=values["data"]
        if not dataFile.endswith(".csv"):
            dataFile=""
            open_window("Please Choose a csv file","ERROR")
            window["data"].update("")
    elif event == "file":  # A file was chosen from the listbox
        path=values["file"]
        if not path.endswith(".sav"):
            path=""
            window["noValid"].update("No Valid Model (.sav) Loaded Yet", text_color="Red")
        else:
            window["noValid"].update(path.split('/')[-1]+" Has Been Loaded", text_color="LightGreen")
    elif event == "save":
        window["saving"].update("Saving...")
        if not values["save"]=="":
            learn(values["save"],values["arabic"],values["url"],values["number"],values["punct"],values["stopw"],values["dia"],values["norm"],values["Stem"],values["Class"],values["split"])

            # t1 = threading.Thread(target=learn, args=(values["save"],values["arabic"],values["url"],values["number"],values["punct"],values["stopw"],values["dia"],values["norm"],values["Stem"],values["Class"],values["split"],))
            # t1.start()
            # t1.join()
            # window["saving"].update("")
            # window.refresh()
        window["save"].update("")
window.close()






