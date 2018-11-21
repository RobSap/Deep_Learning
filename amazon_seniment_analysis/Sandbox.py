from __future__ import print_function
import os
import sys
from operator import add
from pyspark.sql import SparkSession
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.sql import SQLContext
from pyspark import SparkContext
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import wordcloud
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import time
from collections import Counter
from apyori import apriori
#list1=['apple','egg','apple','banana','egg','apple']
#counts = Counter(list1)
#print(counts)

def predctions_helpful_and_not(df,word):
    df2=convert_spark_df_to_pd_df(df)
    #print(df2)
    
    if word =="helpful":
        df2['vote'] = [1 if x[0] > 3 else 0 for x in (df2.helpful)]
    else:
        df2['vote'] = [1 if x[1] > 3 else 0 for x in (df2.helpful)]
    
    word_cloud(df2,word)
    #print("DF2") 
    #print(df2) 
    review_text = df2['reviewText']

    #print("review text")
    #print(review_text)
   

    x_train, x_test, y_train, y_test = train_test_split(df2.reviewText, df2.vote, random_state=0)

    #print(x_train)
    #Count words, only add them if they have a min of 3
    #Used for text feature extraction (from sklearn)
    #Sklearn: the number of features will be equal to the vocabulary size found by analyzing the data.
    #CountVectorizer works like a bad of words, we assign each new word to a id
    # we then get a count for each word (and use the rating as the label)
    vectorizer = CountVectorizer(min_df=3).fit(x_train)
    X_train = vectorizer.transform(x_train)
    X_test = vectorizer.transform(x_test)

    #print(X_train)
    feature_names = vectorizer.get_feature_names()
    #print(feature_names)

    lrmodel =LogisticRegression()
    fitmodel =lrmodel.fit(X_train, y_train)
    results = fitmodel.predict(X_test)
    
    
    scores = accuracy_score(y_test, results)

    print("LR")
    print(scores)


    model = GradientBoostingClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    results2 = model.predict(X_test)
    accuracy = accuracy_score(y_test, results2)

    print("GBC")
    print(accuracy) 

    model3 = RandomForestClassifier(n_estimators=10)
    model3.fit(X_train, y_train)
    results3 = model3.predict(X_test)
    accuracy2 = accuracy_score(y_test, results3)
  
    print("RF") 
    print(accuracy2)



def predctions_good_bad_review(df,word):
    
    print("Calculating " + str(word) + " words")
    df2=convert_spark_df_to_pd_df(df)
    if word =="good":
        df2['positive_and_negative'] = [1 if x > 3 else 0 for x in (df2.overall)]
    else:
        df2['positive_and_negative'] = [1 if x <= 3 else 0 for x in (df2.overall)]
    #print("DF2") 
    #print(df2) 
    review_text = df2['reviewText']

    #print("review text")
    #print(review_text)
    x_train, x_test, y_train, y_test = train_test_split(df2.reviewText, df2.positive_and_negative, random_state=0)

    #Count words, only add them if they have a min of 3
    #Used for text feature extraction (from sklearn)
    #Sklearn: the number of features will be equal to the vocabulary size found by analyzing the data.
    #CountVectorizer works like a bad of words, we assign each new word to a id
    # we then get a count for each word (and use the rating as the label)
    vectorizer = CountVectorizer(min_df=3).fit(x_train)

    #print("x_train")
    #print(x_train)  
    X_train = vectorizer.transform(x_train)
    X_test = vectorizer.transform(x_test)

    #print(X_train)
    feature_names = vectorizer.get_feature_names()

    start = time.time()

    #limit = 9282
    limit = 1000000
    #limit = 1000
    #limit = 100
    lrmodel =LogisticRegression()
    fitmodel =lrmodel.fit(X_train[:limit], y_train[:limit])
    results = fitmodel.predict(X_test)
    scores = accuracy_score(y_test, results)
    
    print("\nLR")
    print(scores)
    print("Run time")
    print(time.time()-start)  

    start = time.time()
    model = GradientBoostingClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    results2 = model.predict(X_test[:limit])
    accuracy = accuracy_score(y_test[:limit], results2[:limit])
    
    print("\nGBC")
    print(accuracy) 
    print("Run time")
    print(time.time()-start)  

    start = time.time()
    model3 = RandomForestClassifier(n_estimators=10)
    model3.fit(X_train, y_train)
    results3 = model3.predict(X_test[:limit])
    accuracy2 = accuracy_score(y_test[:limit], results3[:limit])
  
    print("\nRF") 
    print(accuracy2)
    print("Run time")
    print(time.time()-start)

    start = time.time()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
    clf.fit(X_train, y_train)
    results3 = model3.predict(X_test[:limit])
    accuracy2 = accuracy_score(y_test[:limit], results3[:limit]) 
    print("\nMLP")
    print(accuracy2)
    print("Run time")
    print(time.time()-start)

def convert_spark_df_to_pd_df(df):
    return df.toPandas()

def convert_pd_df_to_spark_df(df,spark):
    return spark.createDataFrame(df)

def word_cloud(df,word):
    #Remove common words like the, there, than and other stop words
    #Also remove most popular words from 1 category
    my_stop_words = STOPWORDS | ENGLISH_STOP_WORDS |{"think"}|{"thing"}|{"serie"}|{"series"} |{"season"}|{"show"}|{"good"}| {'really'}|{'make'}|{'people'}|{'characters'}|{"character"}|{"episode"}|{"film"}|{"movie"}|{"show"}|{"episode"}
    cloud = wordcloud.WordCloud(stopwords=my_stop_words,background_color='white',max_words=200, max_font_size=60,relative_scaling=1).generate(' '.join((df).reviewText))
    fig = plt.figure(1)
    plt.imshow(cloud);
    plt.axis('off')
    #plt.show()
    fig.savefig(word+".png",dpi=900)


def run_spark(spark, load):
    
    df = spark.read.json(load)
    #df.show()
    df=df.drop('helpful','reviewerID','asin','reviewerName','reviewTime','unixReviewTime')
    #df=df.drop('reviewerID','reviewerName','reviewTime','unixReviewTime')
    #df.show()

    return df


def main():
    spark = SparkSession\
        .builder\
        .appName("Python_Amaozon")\
        .getOrCreate()
    #Load all the files in one directory, return df
    print("Loading json file(s)")
    #df=run_spark(spark,'/Volumes/TOSHIBA/desktop/data_sets/')
    df=run_spark(spark,'/Volumes/TOSHIBA/desktop/data_sets/reviews_Amazon_Instant_Video_5.json')
    #reviewText
    df2=convert_spark_df_to_pd_df(df) 

    print(df2)
    df2=pd.DataFrame(columns=['reviewText'])


    #Data Mining
    ''' 
    print(df2)
    results = list(apriori(df2))

    print("results from aprior")
    print(results)
    '''


    '''
    print(Counter(" ".join(df2["reviewText"]).split()).most_common(1000)) 

    myWords = Counter(" ".join(df2["reviewText"]).split()).most_common(1000)

    my_stop_words = STOPWORDS | ENGLISH_STOP_WORDS |{"think"}|{"thing"}|{"serie"}|{"series"} |{"season"}|{"show"}|{"good"}| {'really'}|{'make'}|{'people'}|{'characters'}|{"character"}|{"episode"}|{"film"}|{"movie"}|{"show"}|{"episode"} 

    for w in myWords:
        if w[0] not in my_stop_words:
            print(w)
    '''
    #Make a word cloud for fun
    #word_cloud(convert_spark_df_to_pd_df(df),"all_words")

    #Make predictions based on word count. Add up the value of each word. And based on the fequency
    #make the prediciton if the review is positive or negative
 
    predctions_good_bad_review(df,"good")
   
    ''' 
    #Same preidctions as above, but predictiong bad
    predctions_good_bad_review(df,"bad")
    '''

    '''
    
     #Make a word gram for the helpful words
    df2=df.drop('asin','helpful','summary')
    helpful_df = df2[df2['overall'].between(4.0, 5.0)]
    #helpful_df.show()
    word_cloud(convert_spark_df_to_pd_df(helpful_df),"helpful_words")
    '''

    '''
    #Make a word gram for the none helpful words
    not_helpful_df = df2[df2['overall'].between(1.0, 2.0)]
    word_cloud(convert_spark_df_to_pd_df(not_helpful_df),"no_helpful_words")
    #not_helpful_df.show()
    '''

    '''
    #print("Running helpfull")
    #Do users find the reviews help full?
    #Can we base this off word choice ?
    predctions_helpful_and_not(df,"helpful")
  
    #print("Running not helpfull")
    predctions_helpful_and_not(df,"not_helpful")

    '''
   
    ''' 
    #Load only amazon instant video results
    df_small=run_spark(spark,"data_sets/reviews_Amazon_Instant_Video_5.json")

    temp0= df.select('asin').collect() 
    temp = df.select('reviewText').collect()
    temp2 = df.select('overall').collect()
    temp3 = df.select('helpful').collect()
    #df=df.count().orderBy('count', ascending=False)
    print(df)
    '''

    '''
    print("Using Sentiment analyzer") 
    for i,item in enumerate(temp):
        if ((temp3[i].helpful[0] >3 or temp3[i].helpful[1] >  3)):# and temp0[i].asin=='B00I3MPDP4'):
            print("Helpfull votes: " + str(temp3[i].helpful[0]))
            print("UnHelpfull votes: " + str(temp3[i].helpful[1]))
            
            print("On review " + str(i) + " of " + str(len(temp)))
            print(temp[i].reviewText)
            blob = TextBlob(temp[i].reviewText,analyzer=NaiveBayesAnalyzer())
            print("Sentiment")
            print(blob.sentiment)
            print("Reviewers actually score: " + str(temp2[i].overall))
            print("Reviewers adjusted score: " + str((25*temp2[i].overall)-25))
            print("Difference in score : " + str((100*blob.sentiment.p_pos) - ((25*temp2[i].overall)-25)))
            print("\n") 
            #predctions_good_bad_review(df,"good")      
    '''



    spark.stop()

if __name__ == "__main__":
    main()


