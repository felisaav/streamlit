#------------------------------
#------import libraries--------
#------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
#import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score

#------------------------------
# read data from email.csv file
#------------------------------
data = pd.read_csv('emails.csv')


#------------------------------
#---Create naive bayes model---
#------------------------------
	#Create a lenght column with the len of emails
data['length'] = data['text'].map(lambda text: len(str(text)))

	#Tokenize: create words from sentences, and removes punctuation
tokenizer = RegexpTokenizer(r'\w+')
data['tokens'] = data.apply(lambda x: tokenizer.tokenize(x['text']), axis = 1)

	#Elimination of stop words
stop=stopwords.words('english')
stop.append('Subject')
data['tokens'] = data['tokens'].apply(lambda x: [item for item in x if item not in stop])

	#Stemming
stemmer = PorterStemmer()
data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

	#Unify the strings once again
data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

	#create train/test split
X_train, X_test, Y_train, Y_test = train_test_split(data['tokens'], 
                                                    data['spam'],
                                                    test_size= 0.2,
                                                    random_state=0)
	#vectorizing 
vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)

	#create naive bayes model
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, Y_train)
predictions = model.predict(vectorizer.transform(X_test))

	#calculate accuracy of the model
accuracy=100 * sum(predictions == Y_test) / len(predictions)
cm = confusion_matrix(Y_test, predictions)

	#function predict spam/not spam emails
def predict_category(s, model=model):
    pred = model.predict(vectorizer.transform([s]))
    return pred


#---------------------------
#---Create charts for EDA---
#---------------------------
df2=data.groupby('spam').count().reset_index().replace(0,"not spam").replace(1,"spam")

	#1st pie chart with distribution
fig1 = px.pie(df2,
             values='text',
             names='spam',
             title='Distribution of spam/not spam emails',
             hover_data=['text'], labels={'text':'# of cases'})

	



	#3th distribution of lenght of spam / not spam emails


fig3=px.imshow(cm,
	       text_auto=True)


#---------------------------
#--Configuration of pages---
#---------------------------
def main():
	st.title("Project - email spam analysis")

	menu = ["Problem description","Descriptive Analysis","Predictive Model","Results","About Me"]
	choice = st.sidebar.selectbox("Menu",menu)


	if choice == "Problem description":
		st.subheader("Problem Description")
		st.markdown("""---""")
		# Columns/Layout
		col1,col2 = st.columns(2)

		with col1:
			st.write("***Description***")
			st.write("In this site we will review the analysis performed to a email.csv file.")
			st.write("The dataset contains different email texts with a classification if it is spam or not.")
			st.write("The objetive of this analysis is to show a review of the dataset, one possible way to solve \
					this kind of problems and the main results")
			st.write("Let's start!")
		# Results Layouts
		with col2:
			st.write("***Original dataset***")
			st.write("spam emails examples")
			st.dataframe(data.loc[data["spam"]==1]["text"].head())
			st.markdown("---")
			st.write("not spam emails examples")
			st.dataframe(data.loc[data["spam"]==0]["text"].head())



	elif choice == "Descriptive Analysis":
		st.subheader("Descriptive Analysis")
		st.markdown("""---""")
		
		st.plotly_chart(fig1)



		st.write('I hate windows')
		st.write(pd.DataFrame({
 			'first_column':[1,2,3,4],
  			'second_column':[10,20,30,40]
		}))

	elif choice == "Predictive Model":
		st.subheader("Predictive Model")
		st.write('I love windows')

	elif choice == "Results":
		st.subheader("Results")
		st.markdown("""---""")
		st.pyplot(fig3)
					
	else:	
		st.subheader("About Me")
		st.write('yo amo windows')





if __name__ == '__main__':
	main()
