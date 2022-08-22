import pyttsx3 #pip install pyttsx3
import datetime
import wikipedia #pip install wikipedia
import webbrowser
import os
  
    
    
from tkinter import *
import numpy as np
import pandas as pd

import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score

 

from sklearn import*
import seaborn as sns



engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice', voices[1].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")

    elif hour>=12 and hour<16:
        speak("Good Afternoon!")   

    else:
        speak("Good Evening!")  

    speak(".............I am karen your healthcare assistent...........,i am here to help you......... ,kindly fill your Name and Symptom........................,so i can predict............... if you having any disease")  


wishMe()



# from gui_stuff import *
#symptoms
l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv(r'C:\Users\ADMIN\Downloads\Disease-prediction-using-Machine-Learning-master\Training.csv')

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

#print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
#print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv(r'C:\Users\ADMIN\Downloads\Disease-prediction-using-Machine-Learning-master\Testing.csv')
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------

def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    #print(accuracy_score(y_test, y_pred))
    score_svm = round(accuracy_score(y_test, y_pred)*100,2)
    print("The accuracy score achieved using decision tree is: "+str(score_svm)+" %")
    #print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
        k=disease[a]
        n=Name.get()
        speak("hi.........."+n+"................According to decision tree algorithm you have"+k+"diease")
        r=wikipedia.summary(k,sentences=1)
        speak("According to wikipedia")
        speak(r)
        #print(r)
        #open disease in webbrowser
        webbrowser.open(r)  
        
        t4.delete("1.0",END)
        t4.insert(END,r)
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")
  
    print(k)
    
      
    t4.delete("1.0",END)
    t4.insert(END,r)
    
def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    score_svm = round(accuracy_score(y_test, y_pred)*100,2)
    #print(accuracy_score(y_test, y_pred))
    print("The accuracy score achieved using randomForest is: "+str(score_svm)+" %")
    
    #print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
        l=disease[a]
        n=Name.get()
        speak("hi.........."+n+"................According to random forest algorithm you have"+l+"diease")
        r=wikipedia.summary(l,sentences=1)
        speak("According to wikipedia..........")
        speak(r)
        #print(r)
        
        webbrowser.open(r)
        
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")
    print(l)
    
   
    t4.delete("1.0",END)
    t4.insert(END,r)

    
def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    
    score_svm = round(accuracy_score(y_test, y_pred)*100,2)
    #print(accuracy_score(y_test, y_pred))
    print("The accuracy score achieved using naivebayes is: "+str(score_svm)+" %")
    
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
        m=disease[a]
        n=Name.get()
        speak("hi.........."+n+"................According to Naivebayes algorithm you have"+m+"diease")
   
        r=wikipedia.summary(m,sentences=1)
        speak("According to wikipedia")
        speak(r)
        #print(r)
        
        webbrowser.open(r)
        
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")
    print(m)
   
    t4.delete("1.0",END)
    t4.insert(END,r)
    
    
    
#gui import.......   
    
import io
import base64

try:
    from Tkinter import Tk, Label, Entry, Toplevel, Canvas
except ImportError:
    from tkinter import Tk, Label, Entry, Toplevel, Canvas

from PIL import Image, ImageDraw, ImageTk, ImageFont

   
    
    
# gui_stuff------------------------------------------------------------------------------------

root = Tk()
root.configure(background='white')

file=r"C:\Users\ADMIN\Pictures\Saved Pictures\248464.jpg"
image = Image.open(file)

width, height = image.size
root.resizable(width=1350, height=1080)
root.geometry("%sx%s"%(width, height))

draw = ImageDraw.Draw(image)

photoimage = ImageTk.PhotoImage(image)
Label(root, image=photoimage).place(x=0,y=0)



# entry variables
Symptom1 = StringVar()
Symptom1.set('select Symtom1')
Symptom2 = StringVar()
Symptom2.set('select Symtom2')
Symptom3 = StringVar()
Symptom3.set('select Symtom3')
Symptom4 = StringVar()
Symptom4.set('select Symtom4')
Symptom5 = StringVar()
Symptom5.set('select Symtom5')
Name = StringVar()
Name.set(' ')


# Heading
w2 = Label(root, justify=LEFT, text="DISEASE PREDICTION SYSTEM", fg="black",bg="white")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=0, columnspan=2, padx=100)
w2 = Label(root, justify=LEFT, text='USING MACHINE LEARNING  ', fg="black", bg="white")
w2.config(font=("Aharoni", 30))
w2.grid(row=2, column=0, columnspan=2, padx=100)

# labels
NameLb = Label(root, text="Name of the Patient", fg="black", bg="white")
NameLb.grid(row=6, column=0, pady=15, sticky=W)


S1Lb = Label(root, text="Symptom 1", fg="black", bg="white")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="black", bg="white")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="black", bg="white")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="black", bg="white")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="black", bg="white")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)


lrLb = Label(root, text="DecisionTree", fg="black", bg="white")
lrLb.grid(row=15, column=0, pady=10,sticky=W)

destreeLb = Label(root, text="RandomForest", fg="black", bg="white")
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="NaiveBayes", fg="black", bg="white")
ranfLb.grid(row=19, column=0, pady=10, sticky=W)


wikiLb = Label(root, text="About Disease ", fg="black", bg="white")
wikiLb.grid(row=21, column=0, pady=10, sticky=W)

# entries
OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)

#buttons
dst = Button(root, text="DecisionTree", command=DecisionTree,bg="gray",fg="skyblue")
dst.grid(row=8, column=3,padx=10)

rnf = Button(root, text="Randomforest", command=randomforest,bg="gray",fg="skyblue")
rnf.grid(row=9, column=3,padx=10)

lr = Button(root, text="NaiveBayes", command=NaiveBayes,bg="gray",fg="skyblue")
lr.grid(row=10, column=3,padx=10)


#textfileds
t1 = Text(root, height=2, width=40,bg="white",fg="gray")
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=2, width=40,bg="white",fg="gray")
t2.grid(row=17, column=1 , padx=10)

t3 = Text(root,height=2, width=40,bg="white",fg="gray")
t3.grid(row=19, column=1 , padx=10)

t4 = Text(root,height=7, width=130,bg="white",fg="gray")
t4.grid(row=21, column=1 , padx=10)

root.mainloop()


