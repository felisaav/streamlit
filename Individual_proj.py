#------------------------------
#------import libraries--------
#------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from collections import Counter

#------------------------------
#----------functions-----------
#------------------------------
# read data from email.csv file
@st.cache_data 
def load_data(url):
    df = pd.read_csv(url)
    return df

# transform dataframe
@st.cache_data
def transform(df):
	#Create a lenght column with the len of emails
	df['length'] = df['text'].map(lambda text: len(str(text)))
	
	#Tokenize: create words from sentences, and removes punctuation
	tokenizer = RegexpTokenizer(r'\w+')
	df['tokens'] = df.apply(lambda x: tokenizer.tokenize(x['text']), axis = 1)

	#Elimination of stop words
	stop=stopwords.words('english')
	stop.append('Subject')
	df['tokens'] = df['tokens'].apply(lambda x: [item for item in x if item not in stop])

	#Stemming
	stemmer = PorterStemmer()
	df['tokens_lst'] = df['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

	#Unify the strings once again
	df['tokens'] = df['tokens_lst'].apply(lambda x: ' '.join(x))
	
	return df
	
#---train naive bayes model---
@st.cache_data
def train_model(X_train, Y_train):
	vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
	X_train_vectorized = vectorizer.transform(X_train)
	#train naive bayes model
	model = MultinomialNB(alpha=0.1)
	model.fit(X_train_vectorized, Y_train)
	return model, vectorizer

@st.cache_data
def word_counter(df):
	# Tokenize and count word frequencies
	word_freq = Counter()
	for text in df['tokens']:
	    tokens = word_tokenize(text)
	    word_freq.update(tokens)
	
	# Convert the word frequencies to a DataFrame
	word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index', columns=['Frequency'])
	
	# Sort the DataFrame by frequency in descending order
	word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)
	word_freq_df = word_freq_df.reset_index()
	# Display the DataFrame with word frequencies
	return(word_freq_df)


#------------------------------
#---------run the code---------
#------------------------------
#load data
data = load_data('emails.csv')
data=transform(data)

#create train/test split
X_train, X_test, Y_train, Y_test = train_test_split(data['tokens'],data['spam'],test_size= 0.2,random_state=0)

#train & predict
model, vectorizer=train_model(X_train, Y_train)
predictions = model.predict(vectorizer.transform(X_test))

#calculate performance metrics
cm = pd.DataFrame(confusion_matrix(Y_test, predictions))
accuracy=100 * sum(predictions == Y_test) / len(predictions)
precision = cm.iloc[1, 1]/(cm.iloc[0, 1]+cm.iloc[1, 1])*100# / (cm.iloc[0, 1] + cm.iloc[1, 1])) #TP/Predicted positives
specificity = cm.iloc[0, 0]/(cm.iloc[0, 1]+cm.iloc[0, 0])*100 #TN/negatives(TN+FP)
recall= cm.iloc[1, 1]/(cm.iloc[1, 1]+cm.iloc[1, 0])*100  #TP/positives(TP+FN)
	
#---------------------------
#------Create charts--------
#---------------------------
#pie chart with distribution - matplotlib chart
df2=data.groupby('spam').count().reset_index().replace(0,"not spam").replace(1,"spam")
fig1, ax1 = plt.subplots()
ax1.pie(df2['text'], labels=df2['spam'], autopct='%1.1f%%', startangle=90)

#distribution of lenght of spam / not spam emails - matplotlib chart
fig2, ax = plt.subplots()
data[data['spam'] == 0]['length'].plot.hist(bins=50, alpha=0.5, color='blue',density=True, label='spam = 0', ax=ax)
data[data['spam'] == 1]['length'].plot.hist(bins=50, alpha=0.5, color='orange',density=True, label='spam = 1', ax=ax)
ax.set_xlabel('Length')
ax.set_ylabel('Frequency')
#ax.set_title('Distribution of Email Lengths')
ax.legend()

#heatmap confusion matrix with results  - plotly chart
fig3, ax2 = plt.subplots()
heatmap = go.Heatmap(z=cm,
		     x=['Predicted not spam', 'Predicted spam'],
		     y=['Real not spam', 'Real spam'],
		     colorscale='Blues')
ax2 = go.Layout(title='Confusion Matrix')
fig3 = go.Figure(data=[heatmap], layout=ax2)

#more frequent words in spam/not spam emails
spam_words=word_counter(data.loc[data['spam']==1]).head(10)
fig4,ax3 = plt.subplots()
sns.barplot(data=spam_words, y="index", x="Frequency",ax=ax3, color='orange')

not_spam_words=word_counter(data.loc[data['spam']==0]).head(10)
fig5,ax4 = plt.subplots()
sns.barplot(data=not_spam_words, y="index", x="Frequency",ax=ax4, color='blue')

#---------------------------
#--Configuration of pages---
#---------------------------
def main():
	st.title("Project - email spam analysis")
	menu = ["Problem description","Descriptive Analysis","Predictive Model","Results"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Problem description":
		st.subheader("Problem Description")
		st.markdown("""---""")
		st.write('<p style="font-size:24px; color:blue;">Intro</p>',unsafe_allow_html=True)
		st.write('''Email spam is a serious problem that affects millions of Internet users every day. Spam emails are unsolicited messages 
  			that are sent in bulk, usually for commercial purposes. They can be annoying, time-consuming, and potentially harmful to 
     			the recipients.''')
		st.write('''According to Statista, nearly **49%** of all emails worldwide were identified as **spam** in 2022. That 
			means that out of the **306.4 billion** emails sent daily, about **150 billion were spam**.''')
		st.write('''Spam emails can have negative impacts on both businesses and customers. For businesses, spam emails can reduce 
   			productivity, increase costs, damage reputation, and expose them to legal risks. For customers, spam emails can invade 
      			their privacy, waste their bandwidth, expose them to scams, phishing, malware, and ransomware, and harm the environment
	 		by generating carbon emissions.''')
		
		st.write('<p style="font-size:24px; color:blue;">Description</p>',unsafe_allow_html=True)
		st.write('''In this site we will review the analysis performed to a email.csv file. \
  			The dataset contains **5,730 rows** with emails body and a classification if the email is a spam or not. \
     			The objetive of this analysis is to show a review of the dataset, one possible way to solve \
				this kind of problems and the main results. \
    				Below you can see an example of the dataset:''')

		st.write('<p style="font-size:24px; color:blue;">Original Dataset</p>',unsafe_allow_html=True)
		col1,col2 = st.columns(2)
		with col1:
			st.write("spam emails examples")
			st.dataframe(data.loc[data["spam"]==1]["text"].head())
		with col2:
			st.write("not spam emails examples")
			st.dataframe(data.loc[data["spam"]==0]["text"].head())

	elif choice == "Descriptive Analysis":
		st.subheader("Descriptive Analysis")
		st.markdown("""---""")
		col1,col2 = st.columns(2)
		with col1:
			st.write("total dataset (5,730 emails) contains **23.9% of spam** emails", unsafe_allow_html=True)
			st.pyplot(fig1) #matplotlib piechart
		with col2:
			st.write("spam emails have a **right Skewed** shape (vs not spam), showing that in general are **shorter emails**")
			st.pyplot(fig2) #matplotlib chart len distribution
		
		col3,col4 = st.columns(2)
		with col3:
			st.write("***10 most freq words in spam emails***")
			st.pyplot(fig4) #seaborn chart most freq words
		with col4:
			st.write("***10 most freq words in not spam emails***")
			st.pyplot(fig5) #seaborn chart most freq words
		st.write("The first conclusion around the most popular words, is that spam emails have more noisy content, with **-** or letters without specific meaning")
	
	elif choice == "Predictive Model":
		st.subheader("Predictive Model")
		st.markdown("""---""")

		st.image('media/ML_spam.png')
		st.write('''It is important to develop effective methods to classify and filter spam emails from legitimate ones. Email 
   			spam classification is the process of using machine learning algorithms to automatically identify and label spam emails 
      			based on their content, sender, subject, and other features.''')
		st.write('''Email spam classification can help users and organizations 
	 		to protect themselves from the threats of spam and improve their email experience.''')
		st.subheader("Naive Bayes")
		st.write('''Naive Bayes is a classification technique that is based on Bayes’ Theorem with an assumption that all the features \
  			that predicts the target value are independent of each other. It calculates the probability of each class and then pick \
     			the one with the highest probability. It has been successfully used for many purposes, but it works particularly well with \
			natural language processing (NLP) problems.''')
		st.write('''Bayes’ Theorem describes the probability of an event, based on a prior knowledge of conditions that might be related to that event.''')
		st.latex(r'''
  			P(H/E)=\frac{P(E/H)*P(H)}{P(E)}
     			''')
		st.subheader("What makes Naive Bayes a ***Naive*** algorithm?")
		st.write('''Naive Bayes classifier assumes that the features we use to predict the target are independent and do not affect each other. \
  			While in real-life data, features depend on each other in determining the target, but this is ignored by the Naive Bayes classifier.\
     			Though the independence assumption is never correct in real-world data, but often works well in practice. so that it is called ***Naive***.
  			''')
		st.write('''This is a simple and really important concept, let's see why''')
		st.write("In this specific example spam classification model

	elif choice == "Results":
		st.subheader("Results")
		st.markdown("""---""")
		col1,col2 = st.columns([3, 1])
		with col1:
			#st.write("***Confusion Matrix***")
			st.plotly_chart(fig3,use_container_width=True,use_container_height=True)
		with col2:
			#st.dataframe(cm)
			st.write("***Main metrics***")
			st.write(f"Accuracy: {accuracy:.2f}%")
			st.write(f"Precision: {precision:.2f}%")
			st.write(f"Specificity: {specificity:.2f}%")
			st.write(f"Recall: {recall:.2f}%")
		st.markdown("""---""")
		st.subheader("Test your email content and see if it is a spam or not")
		with st.form(key='test_email'):
			msg =st.text_area("write your email here")
			submit_code = st.form_submit_button("Execute")
		if submit_code:
			st.info("Query Result")
			if model.predict(vectorizer.transform([msg]))[0]==1:
				st.write('Your message is a spam')
			else:
				st.write('Your message is a normal email')

if __name__ == '__main__':
	main()
