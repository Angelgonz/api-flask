from asyncore import write
from re import L
import sqlite3
from unicodedata import name
from colorama import Cursor
from flask import Flask, request,jsonify
from config import config
import mysql.connector
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#index=open("datasets/trec07p/full/index").readlines()
#print(angel)
app = Flask(__name__)
#conexion = MySQL(app)

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
nltk.download('stopwords')
#------------
class Parser:

    def __init__(self):


        self.stemmer = nltk.PorterStemmer()
        #print(nltk.PorterStemmer())
        #--o=open("da/corpora/stopwords/english").readlines()
        #print (o)
        #--im=[]
        #---for h in o:
            #--im.append(h.rstrip())
        
        self.stopwords =nltk.corpus.stopwords.words('english') #--set(im)
        print(nltk.corpus.stopwords.words('english'))
        #print(nltk.corpus.stopwords.words('english'))
        self.punctuation = list(string.punctuation)
        

    def parse(self, email_path):
        #Parse an email.
        #with open(email_path, errors='ignore') as e:
         #   msg = email.message_from_file(e)
        #print("hola%ssaasca"%email_path)
        try:
            connection=mysql.connector.connect(
            host='angel.cbk7spjlzjy0.us-east-2.rds.amazonaws.com',
            port=3306,
            user='admin',
            password='avefenix',
            db='angel'
            )
            if connection.is_connected():
                #print("conecion exitosa")
                #info_server=connection.get_server_info()
                #print(info_server)

                cursor=connection.cursor()
                sq="SELECT textto from angel.mail where idemail='%s'"%email_path
                cursor.execute(sq)
                #row=cursor.fetchall()
                for lucero in cursor:
                    nose=" "
                    #msg=email.message_from_binary_file(lucero)
                    #print(msg)
                    nose=lucero
                
                #print(row)
        except Exception as ex:
            print(ex)
        finally:
            if connection.is_connected():
                connection.close()
                #print("conexio terminada")

        #cursor2= conexion.connection.cursor()
        #sq="SELECT textto from angel.mail where idemail='%s'"%email_path
        #cursor2= conexion.cursor()
        #cursor2.execute(sq)
        #cursor2.fetchall()
        #print(cursor2)
        l=""
        #print(cursor2.__str__())
        #for lucero in cursor2:
         #   nose=" "
            #msg=email.message_from_binary_file(lucero)
            #print(msg)
          #  nose=lucero
        #curso2.close()
        #conexion.close()
            
            
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
        #Extract the email content.
        subject = self.tokenize(msg['Subject']) if msg['Subject'] else []
        body = self.get_email_body(msg.get_payload(),
                                   msg.get_content_type())
        content_type = msg.get_content_type()
        # Returning the content of the email
        return {"subject": subject,
                "body": body,
                "content_type": content_type}

    def get_email_body(self, payload, content_type):
        #Extract the body of the email
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
        ##Transform a text string in tokens. Perform two main actions,
        #clean the punctuation symbols and do stemming of the text.
        for c in self.punctuation:
            text = text.replace(c, "")
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        tokens = list(filter(None, text.split(" ")))
        # Stemming of the tokens
        return [self.stemmer.stem(w) for w in tokens if w not in self.stopwords]


##-------------------errrorrr
def parse_index(path_to_index, n_elements):
    
    
    try:
        connection=mysql.connector.connect(
        host='angel.cbk7spjlzjy0.us-east-2.rds.amazonaws.com',
        port=3306,
        user='admin',
        password='avefenix',
        db='angel'
        )
        if connection.is_connected():
            #print("conecion exitosa")
            info_server=connection.get_server_info()
            #print(info_server)

            cursor=connection.cursor()
            cursor.execute(path_to_index)
            index=cursor.fetchall()
        #print(row)
    except Exception as ex:
            print(ex)
    finally:
            if connection.is_connected():
                connection.close()
        #print("conexio terminada")
    
    #cursor= conexion.connection.cursor()   
    #cursor.execute(path_to_index)
    #index=cursor.fetchall()
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
    print(index_path)
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





@app.route('/mail',methods=['GET'])
def listar_Email():
    try:
        
        
        
        sqlfinal="SELECT * from angel.index limit 50"
        # Leemocreate_prep_datasets 12000 correos electrónicos
        X, y = create_prep_dataset(sqlfinal, 50)
        spam=0
        inc=0
        ham=0
        for i in y:
            #print(i)
            inc+=1
            
            if inc>=1 and inc<=45:

                if i=="spam":
                    spam+=1
                else:
                    ham+=1
        dates="hola"
        
        
        
        return jsonify(dates)
    except Exception as ex:
        return jsonify(ex)



def pagina_no_encontrada(error):
    return "<h1>La pagina no existe que intentas encontrear </h1>"

if __name__ == '__main__': 
    app.config.from_object(config['development'])
    app.register_error_handler(404,pagina_no_encontrada)

    app.run()
