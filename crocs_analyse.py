import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Einlesen und Bearbeitung der CSV
def csv_review_filter(csv_file):
    """ 
    Diese Funktioniert nimmt als Input die CSV-Datei 
    und sucht in dieser nach 1.0 oder 2.0 Bewertungen.
    Anschließend werden von diesen Bewertungen nur das Textfeld zurückgegeben.
    """
    bad_reviews=[]
    with open(csv_file,"r",encoding="utf-8") as csvreview:
        csv_read = csv.reader(csvreview)
        for i in csv_read:
            if i[3] == "1.0" or i[3] == "2.0":
                bad_reviews.append(i[1])
    return bad_reviews


# 2. Bereinung der Daten
# 2.1 Entfernen von Satz-, Sonder- und Zahlenzeichen sowie Kleinschreibung aller Wörter
def remove_bad_char(review):
    """ 
    Dies Funktion nimmt ein Array mit Strings als Input 
    und entfernt alles Zeichen, welche nicht alphabetisch sind.
    Anschließend werden alle Zeichen in Kleinbuchstaben an in einen 
    Array hinzufügt, welches zurückgegeben wird.
    """
    clean_reviews=[]
    for i in review:
        wort=""
        for j in i:
            if j.isalpha() is True or j == " ":
                wort+=j
        clean_reviews.append(wort.lower())
    return clean_reviews


# 2.2 Stoppwörter
def remove_stopwords(review):
    """
    Diese Funktioniert erhält als Input ein Array welcher nur alphabetische Zeichen enthält 
    und schaut, ob sich Stoppwörter in diesen befinden.
    Diese Stoppwörter werden entfernt und schließend wird ein bereinigtes Array zurückgegeben.
    """
    stop_words = set(stopwords.words('english'))
    # Erstellung neuer Stoppwörter, um "'" zu entfernen und Wörter hinzuzufügen.
    new_stopwords = []
    for word in stop_words:
        if "'" in word:
            new_stopwords.append(word.replace("'",""))
        else:
            new_stopwords.append(word)
    new_stopwords.extend(["actually","ago","also",
                        "another","even","definitely","ive","im","theyer","would","unfortunately"])
    clean_review=[]
    for entry in review:
        clean_words=""
        words = entry.split(" ")
        for word in words:
            if word not in new_stopwords:
                clean_words+=word+" "
        clean_review.append(clean_words[:-1])
    return clean_review


# 2.3 Lemmatisierung der Wörter
def lemmatization_words(review):
    """ 
    Bei dieser Funktioniert wird für jedes Wort geschaut, 
    ob sich dieses als Nomen, Verb oder Adjektiv lemmatisieren lässt.
    Anschließend wird ein neues Array zurückgegeben
    """
    clean_review=[]
    lem = WordNetLemmatizer()
    for entry in review:
        clean_words=""
        words = entry.split(" ")
        for word in words:
            c_word=lem.lemmatize(word,"n")
            c_word=lem.lemmatize(c_word,"v")
            c_word=lem.lemmatize(c_word,"a")
            clean_words+=c_word+" "
        clean_review.append(clean_words[:-1])
    return clean_review


# 3. Kovertierung in numerische Vektoren
# 3.1 Bag Of Words und Dicontary
def create_bow(review):
    """
    Hier wird mithilfe der SKLearn Bibliothek ein Bag of Words erstellt.
    Dabei werden nur Wörter betrachtet, der Dokumentenhäufigkeit mindestes bei 12 liegt.
    Anschließend wird das Bow, sowie das dazugehörige Dictionary, zurückgegeben.
    """
    vectorizer = CountVectorizer(min_df=12)
    bow = vectorizer.fit_transform(review)
    dic = vectorizer.get_feature_names_out()
    bow_dataframe = pandas.DataFrame(bow.toarray(),columns=dic)
    bow_dataframe.to_csv("bow.csv")
    return bow, dic


# 3.2 Create TF-IDF und Dicontary
def create_tfidf(review):
    """
    Hier wird mithilfe der SKLearn Bibliothek eine TF-IDF Matrix erstellt.
    Dabei werden nur Wörter betrachtet, der Dokumentenhäufigkeit mindestes bei 12 liegt.
    Mithilfe der Pandas Bibliothek wird das Ergebnis als CSV abgespeichert.
    Anschließend wird die TF-IDF Matrix zurückgegeben.
    """
    vectorizer = TfidfVectorizer(min_df=12)
    tfidf = vectorizer.fit_transform(review)
    tfidf_dataframe = pandas.DataFrame(tfidf.toarray(),columns=vectorizer.get_feature_names_out())
    tfidf_dataframe.to_csv("tfidf.csv")
    return tfidf


# 4. Thema Extrahieren
# 4.1 LSA
def lsa_analysis(vector,dictionary,review):
    """
    Als Input erhält diese Funktion ein Vector Model (Bow oder TF-IDF), 
    das Vokabular und die Reviews.
    Die SKLearn Bibliothek wird genutzt, um ein LSA zu erstellen 
    und die Pandas Bibliothek um dies zu visualisieren.
    Es werden die wichtigsten Themen / Wörter ausgegeben
    und die Ergebnisse als CSV gespeichert.
    Anschließend wird die LSA zurückgegeben. 
    """
    lsa_model = TruncatedSVD(n_components=4)
    lsa = lsa_model.fit_transform(vector)
    lsa_dataframe = pandas.DataFrame(lsa, columns=["Thema1","Thema2","Thema3","Thema4"])
    lsa_dataframe["Review"]=review
    lsa_dataframe[["Review","Thema1","Thema2","Thema3","Thema4"]].to_csv("lsa_review.csv")
    for i in range(4):
        print("LSA: Wichtige Bewertungen zum Thema"+str(i+1)+":")
        print(lsa_dataframe[["Review","Thema"+str(i+1)]].sort_values(by="Thema"+str(i+1),ascending=False)[:5])
        print(" ")

    #Musst du dir nochmal genauer anschauen das T ist um die Tabelle umzudrehen
    word_lda_dataframe = pandas.DataFrame(lsa_model.components_,index=["Thema1","Thema2","Thema3","Thema4"],columns=dictionary).T
    word_lda_dataframe.to_csv("lsa_words.csv")
    for i in range(4):
        print("LSA: Wichtige Wörter für Thema "+str(i+1))
        print(word_lda_dataframe[["Thema"+str(i+1)]].sort_values(by="Thema"+str(i+1),ascending=False)[:5])
        print(" ")
    print(" ")
    return lsa


# 4.2 LDA und Coherence Score
def lda_analysis(review):
    """
    In dieser Funktion wird als erstes ein neues Dictionary, BOW  und ID2Word erstellt.
    Anschließend werden 8 LDA Modelle erstellt und 
    von diesen der Cohrence Score berechnet und als CSV gespeichert.
    Da in diesem Beispiel die Anzahl 3 als guter Wert herausgekommen ist, 
    wird anschließend ein SDA Model mit 3 Themen aufgestellt.
    Die Ergebnisse dieses Models werden als CSV gespeichert,
    die wichtigsten Thema ausgegeben und anschließend wird das Model zurückgegeben.
    """
    word_array = []
    for r in review:
        word_array.append(r.split())
    dic = Dictionary(word_array)
    dic.filter_extremes(no_below=12)
    dbow = [dic.doc2bow(words) for words in word_array]
    load_dictinoary = dic[0] # Wird benötigt um das Dictionary zu laden
    id2word = dic.id2token
    coherence_score={}
    for i in range(2,11):
        lda_model=""
        lda_model = LdaModel(
            corpus=dbow,
            id2word=id2word,
            num_topics=i,
        )
        coherence_model = CoherenceModel(model=lda_model,corpus=dbow,coherence='u_mass')
        coherence_score[i] = coherence_model.get_coherence()

    coherence_score_dataframe = pandas.DataFrame()
    coherence_score_dataframe["Anzahl Themen"] = (coherence_score.keys())
    coherence_score_dataframe["Score"] = (coherence_score.values())
    coherence_score_dataframe.to_csv("coherence_score.csv",index=False)
    lda_topmodel = LdaModel(
        corpus=dbow,
        id2word=id2word,
        num_topics=3,
    )
    lda_words = lda_topmodel.get_topics()
    lda_words_dataframe = pandas.DataFrame(lda_words,index=["Thema1","Thema2","Thema3"], columns=dic.values()).T
    lda_words_dataframe.to_csv("lda_words.csv")

    for i in range(3):
        print("LDA: Wichtige Wörter für Thema "+str(i+1))
        print(lda_words_dataframe[["Thema"+str(i+1)]].sort_values(by="Thema"+str(i+1),ascending=False)[:5])
        print(" ")
    return lda_topmodel


bad_crocs_reviews=csv_review_filter("croc_reviews.csv")

cleanchar_bad_crocs_reviews=remove_bad_char(bad_crocs_reviews)

cleanwords_bad_crocs_reviews=remove_stopwords(cleanchar_bad_crocs_reviews)

lemword_bad_crocs_reviews=lemmatization_words(cleanwords_bad_crocs_reviews)

crocs_bow,crocs_dic = create_bow(lemword_bad_crocs_reviews)

crocs_tfidf = create_tfidf(lemword_bad_crocs_reviews)

crocs_lsa = lsa_analysis(crocs_tfidf,crocs_dic,lemword_bad_crocs_reviews)

crocs_lda = lda_analysis(lemword_bad_crocs_reviews)