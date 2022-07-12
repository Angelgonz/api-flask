#from asyncore import write
#from re import L
#import sqlite3
#from unicodedata import name
#from colorama import Cursor
from flask import Flask, request,jsonify
from config import config
from flask_mysqldb import MySQL
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#index=open("datasets/trec07p/full/index").readlines()
#print(angel)
conexion = MySQL(app)

#----------------inicia el codigo del machine

# Esta clase facilita el preprocesamiento de correos electrónicos que poseen código HTML
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)

##-------------------------------

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


#----------------
t = '<tr><td align="left"><a href="../../issues/51/16.html#article">Phrack World News</a></td>'
strip_tags(t)
#---------------------------------
import email
import string
import nltk
#nltk.download('stopwords',download_dir='./da/')
#------------
class Parser:

    def __init__(self):


        self.stemmer = nltk.PorterStemmer()
        #print(nltk.PorterStemmer())
        o=open("da/corpora/stopwords/english").readlines()
        #print (o)
        im=[]
        for h in o:
            im.append(h.rstrip())
        
        self.stopwords = set(im)
        #print(nltk.corpus.stopwords.words('english'))
        #print(nltk.corpus.stopwords.words('english'))
        self.punctuation = list(string.punctuation)
        

    def parse(self, email_path):
        """Parse an email."""
        #with open(email_path, errors='ignore') as e:
         #   msg = email.message_from_file(e)
        #print("hola%ssaasca"%email_path)
        sq="SELECT textto from angel.mail where idemail='%s'"%email_path
        cursor2= conexion.connection.cursor()
        cursor2.execute(sq)
        cursor2.fetchall()
        #print(cursor2)
        l=""
        #print(cursor2.__str__())
        for lucero in cursor2:
            nose=" "
            #msg=email.message_from_binary_file(lucero)
            #print(msg)
            nose=lucero
            
            
        #print(nose)
        #print(" ")
        for i in nose:
            l+=i
        #print(l)
        msg=email.message_from_string(l)
        #print(nu)
        
        #print(msg)
        return None if not msg else self.get_email_content(msg)

    def get_email_content(self, msg):
        """Extract the email content."""
        subject = self.tokenize(msg['Subject']) if msg['Subject'] else []
        body = self.get_email_body(msg.get_payload(),
                                   msg.get_content_type())
        content_type = msg.get_content_type()
        # Returning the content of the email
        return {"subject": subject,
                "body": body,
                "content_type": content_type}

    def get_email_body(self, payload, content_type):
        """Extract the body of the email."""
        body = []
        if type(payload) is str and content_type == 'text/plain':
            return self.tokenize(payload)
        elif type(payload) is str and content_type == 'text/html':
            return self.tokenize(strip_tags(payload))
        elif type(payload) is list:
            for p in payload:
                body += self.get_email_body(p.get_payload(),
                                            p.get_content_type())
        return body

    def tokenize(self, text):
        """Transform a text string in tokens. Perform two main actions,
        clean the punctuation symbols and do stemming of the text."""
        for c in self.punctuation:
            text = text.replace(c, "")
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        tokens = list(filter(None, text.split(" ")))
        # Stemming of the tokens
        return [self.stemmer.stem(w) for w in tokens if w not in self.stopwords]


##-------------------errrorrr
def parse_index(path_to_index, n_elements):
    cursor= conexion.connection.cursor()
    
        
    cursor.execute(path_to_index)
    index=cursor.fetchall()
        
        
    #print(index[1][2])
    ret_indexes = []
    for i in range(n_elements):
        #mail = index[i].split(" ../")
        #label = mail[0]
        #path = mail[1][:-1]
        #ret_indexes.append({"label":label, "email_path":os.path.join(DATASET_PATH, path)})
        mail = index[i]
        label = mail[1]
        path = mail[2][:-1]
        #print("label %s mail %sa"%(label,path))
        #print("-------------------esti es el path que devuelve loco %s   --------------"%ret_indexes)
        ret_indexes.append({"label":label, "email_path":path})
    return ret_indexes

#--------------------------
def parse_email(index):
    p = Parser()
    pmail = p.parse(index["email_path"])
    return pmail, index["label"]
#---------------------


def create_prep_dataset(index_path, n_elements):
    X = []
    y = []
    indexes = parse_index(index_path, n_elements)
    for i in range(n_elements):
        #print("\rParsing email: {0}".format(i+1), end='')
        mail, label = parse_email(indexes[i])
        X.append(" ".join(mail['subject']) + " ".join(mail['body']))
        y.append(label)
    return X, y



#----------------





@app.route('/mail/',methods=['GET'])
def listar_Email():
    try:
        """"
        #mail="inmail.2"
        #cursor= conexion.connection.cursor()
        sql="SELECT * from angel.index limit 100"
        
        #cursor.execute(sql)
        #index=cursor.fetchall()
        
        
        #print(index[1][2])
        #-----------------
        
        #for i in range(10):
        #    mail = index[i]
        #    label = mail[1]
        #    path = mail[2][:-1]
        #    print("label %s mail %sa"%(label,path))
        #-------
        #d=pd.DataFrame(index)
        #cursor= conexion.connection.cursor()
        #sql="SELECT textto from mail where idemail = '%s'"%mail
        #a=d.loc[0][2]
        #for recibir in d[2,1]:
        #-------obtener los datos por email del index
        #print(a)
        #sql1="SELECT * from angel.index where nom='%s'"%a
        #cursor1= conexion.connection.cursor()
        #cursor1.execute(sql1)
        #nuevo=cursor1.fetchall()
        #print(nuevo)
        #cursor.close()
        #-----errorrrrr----
        
        #inmail = open(nuevo1).read()
        nu="inmail.1"
        #inmail =open(nuevo1).read()
        p = Parser()
        p.parse(nu)
        sql="SELECT * from angel.index limit 100"
        num=100
        #indexes = parse_index(sql, num)
        #indexes
        #print(indexes)
        sql1="SELECT * from angel.index limit 50"
        num2=1
        index = parse_index(sql1,50 )
        #print(index)
        mail, label = parse_email(index[0])
        print("El correo es:", label)
        print(mail)


        #-------obteber los emial por nombre del index
        #sql1="SELECT * from angel.mail where idemail='%s'"%a.rstrip()
        #cursor1= conexion.connection.cursor()
        #cursor1.execute(sql1)
        #nuevo=cursor1.fetchall()
        #print(nuevo)
        
        #cursor.execute(sql)
        #datos=cursor.fetchall()
        #----------------------
        # iniciando prueba
        # Preapración del email en una cadena de texto
        prep_email = [" ".join(mail['subject']) + " ".join(mail['body'])]

        vectorizer = CountVectorizer()
        X = vectorizer.fit(prep_email)

        print("Email:", prep_email, "\n")
        print("Características de entrada:", vectorizer.get_feature_names())
        X = vectorizer.transform(prep_email)
        print("\nValues:\n", X.toarray())
        

        prep_email = [[w] for w in mail['subject'] + mail['body']]

        enc = OneHotEncoder(handle_unknown='ignore')
        X = enc.fit_transform(prep_email)

        print("Features:\n", enc.get_feature_names())
        print("\nValues:\n", X.toarray())
        #----------------


        X_train, y_train = create_prep_dataset(sql, 100)
        X_train

        print(X_train)
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(X_train)


#----------------------------

        print(X_train.toarray())
        print("\nFeatures:", len(vectorizer.get_feature_names()))

#--------------------------

        
        pd.DataFrame(X_train.toarray(), columns=[vectorizer.get_feature_names()])

#----------------

        y_train

#----------------------

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        # Leemos 150 correos de nuestro conjunto de datos y nos quedamos únicamente con los 50 últimos 
        # Estos 50 correos electrónicos no se han utilizado para entrenar el algoritmo
        
        sql4="SELECT * from angel.index limit 150"
        X, y = create_prep_dataset(sql4, 150)
        X_test = X[100:]
        y_test = y[100:]
        #------------------
        #print(X_test)
        X_test = vectorizer.transform(X_test)
        #print("")
        #print(X_test)
#--------------------
        #print("")
        y_pred = clf.predict(X_test)
        y_pred
        #print(y_pred)
#--------------
        #print("Predicción:\n", y_pred)
        print("\nEtiquetas reales:\n", y_test)
        print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
"""

        sqlfinal=sql4="SELECT * from angel.index limit 500"
        # Leemos 12000 correos electrónicos
        X, y = create_prep_dataset(sqlfinal, 500)
        spam=0
        inc=0
        ham=0
        for i in y:
            #print(i)
            inc+=1
            
            if inc>=1 and inc<=450:

                if i=="spam":
                    spam+=1
                else:
                    ham+=1

        #print("spam:  %s y ham:%s"%(spam,ham))

        #print ("datos de entrenamiento")aqsw

        
#---------------------


# Utilizamos 10000 correos electrónicos para entrenar el algoritmo y 2000 para realizar pruebas
        X_train, y_train = X[:450], y[:450]
        X_test, y_test = X[50:], y[50:]
        #print("-------------------este es el train")
        #print("")
        #print(X_train)

        #print("")
        #print("este es la priueba")
        #print(X_test)


#----------------------------

        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(X_train)


#-----------------------------


        clf = LogisticRegression()
        clf.fit(X_train, y_train)

#-------------------------

        X_test = vectorizer.transform(X_test)
#---------------

        y_pred = clf.predict(X_test)

#------------------------
        #print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
#--------------------------------
#----------------------------app.run(threaded=True, port=5000)


#---------------------
        dates={'spam':spam,'ham':ham,'Prediccion':accuracy_score(y_test, y_pred)}

#------------------------

        
        
        return jsonify(dates)
    except Exception as ex:
        return ex



def pagina_no_encontrada(error):
    return "<h1>La pagina no existe que intentas encontrear </h1>"

if __name__ == '__main__':
    app.config.from_object(config['development'])
    app.register_error_handler(404,pagina_no_encontrada)

    app.run(threaded=True, port=80)
