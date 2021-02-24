import pandas as pd                                                             #gia to diavasma ton .csv
import nltk                                                                     #gia pre-processing sinartisis
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer                    #gia tin evresi Tf-Idf enos vector
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection                                             #gia ton diaxorismo ton datasets
from sklearn.model_selection import GridSearchCV                                #gia na velistopiisoume tis parametrous
from sklearn import svm                                                         #gia ti dimiourgia enos Support Vector Machine montelou
from sklearn.svm import SVC                                                     #gia to svm
from sklearn.naive_bayes import MultinomialNB                                   #gia ti dimiourgia tou montelou Naive-Bayes me polionimiki katanomi
from sklearn.metrics import accuracy_score                                      #gia ton prosdiorismo tou accuracy
from sklearn.metrics import classification_report                               #gia to evaluation tou montelou

def initialize_text(df_train):                                                  #sinartisi gia pre-processing ton description
    stemmer = PorterStemmer()                                                   #initialisation tou stemmer os stigmiotipo tis PorterStemmer()
    lemmatizer = WordNetLemmatizer()                                            #initialisation tou lemmatizer os stigmiotipo tis WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))                                #metavliti me periexomeno tis kines leksis tou agklikou leksilogiou
    indexOfDescription = 0

    for description in df_train['Description']:                                 #loop gia tin prospelasi olon ton descriptions sto .csv
        indexOfWord = 0                                                         
        description = word_tokenize(description)                                #tokenization ton descriptions ana leksi
        description = [word.lower() for word in description if word.isalpha()]  #metatropi ton CAPS se mikra ke aferesi ton arithmon
        description = [w for w in description if not w in stop_words]           #aferesi ton kinon lekseon tou agklikou leksilogiou
        for word in description:                                                #loop gia tin prospelasi olon ton lekseon sto description
            description[indexOfWord] = stemmer.stem(word)                       #stemmaroume ti leksi
            indexOfWord = indexOfWord + 1                                       
        df_train.loc[indexOfDescription, 'text_final'] = str(description)       #apothikevoume to processed description se ena neo column onomati "text_final"
        indexOfDescription = indexOfDescription + 1
                
df_train = pd.read_csv(r'/content/JobsDataset.csv')                             #scannaroume to csv kai to apothikevoume sto df_train
initialize_text(df_train)                                                       #kaloume thn sinartisi initialize_text gia to pre-process tou df_train
df_train = df_train.sample(frac=1)                                              #anakatevoume to dataset
x_train, x_test, y_train, y_test = model_selection.train_test_split(df_train['text_final'], df_train['Query'],test_size=0.3) #xorizoume to dataset
from sklearn.preprocessing import StandardScaler
count_vect = CountVectorizer()                                                  #initialisation tou count_vect os stigmiotipo tis CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train)                              #ftiaxetai ena vocabulary me tis lekseis pou iparxoun sta descriptions kai gia 
                                                                                #kathe description dhmiourgite ena vector pou krata plithos emfanisis tis kathe 
                                                                                #leksis se auto  
tfidf_transformer = TfidfTransformer()                                          #initialisation tou tfidf_transformer os stigmiotipo tis TfidfTransformer()
train_vectors = tfidf_transformer.fit_transform(X_train_counts)                 #metatrepei ta values ton vectors se tf-idf

X_test_counts = count_vect.transform(x_test)                                    #kanoume tin idia diadikasia gia to test dataset
test_vectors = tfidf_transformer.transform(X_test_counts)

clf = MultinomialNB().fit(train_vectors, y_train)                               #ftiaxnetai ena modelo tipou naive-bayes gia to train set
predicted = clf.predict(test_vectors)                                           #simfona me to modelo pou ftiaxtike parapano kanoume ena prediction sto test set
print("Naive Bayes Accuracy Score -> ", accuracy_score(predicted, y_test)*100)  #ektiponoume to accracy tou modelou

SVM = svm.SVC(C=1.0, kernel='linear',decision_function_shape='ovo')             #initialisation tou SVM os stigmiotipo tis svm.SVC()
SVM.fit(train_vectors, y_train)                                                 #ftiaxnetai to modelo svm gia to train set
predictions_SVM = SVM.predict(test_vectors)                                     #simfona me to modelo pou ftiaxtike parapano kanoume ena prediction sto test set
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, y_test) * 100)  #ektiponoume to accuracy tou modelou
print(classification_report(y_test,predictions_SVM ))