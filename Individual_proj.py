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
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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
	df['tokens'] = df['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

	#Unify the strings once again
	df['tokens'] = df['tokens'].apply(lambda x: ' '.join(x))
	
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

#------------------------------
#---------run the code---------
#------------------------------
data = load_data('emails.csv')
data=transform(data)

#create train/test split
X_train, X_test, Y_train, Y_test = train_test_split(data['tokens'],data['spam'],test_size= 0.2,random_state=0)

model, vectorizer=train_model(X_train, Y_train)

predictions = model.predict(vectorizer.transform(X_test))

cm = pd.DataFrame(confusion_matrix(Y_test, predictions))

#calculate accuracy of the model
accuracy=100 * sum(predictions == Y_test) / len(predictions)
precision = cm.iloc[1, 1]/(cm.iloc[0, 1]+cm.iloc[1, 1])*100# / (cm.iloc[0, 1] + cm.iloc[1, 1])) #TP/Predicted positives
specificity = cm.iloc[0, 0]/(cm.iloc[0, 1]+cm.iloc[0, 0])*100 #TN/negatives(TN+FP)
recall= cm.iloc[1, 1]/(cm.iloc[1, 1]+cm.iloc[1, 0])*100  #TP/positives(TP+FN)


#class_report=pd.DataFrame(classification_report(Y_test, predictions))

#function predict spam/not spam emails
#@st.cache_data 
#def predict_category(s, model=model):
#    pred = model.predict(vectorizer.transform([s]))
#    return pred
	
#---------------------------
#------Create charts--------
#---------------------------
#1st pie chart with distribution
df2=data.groupby('spam').count().reset_index().replace(0,"not spam").replace(1,"spam")
custom_colors = ['blue', 'orange'] 
fig1 = px.pie(df2,
             values='text',
             names='spam',
             #title='Distribution of spam/not spam emails',
             labels={'text':'# of cases'})
fig1.update_traces(marker=dict(colors=custom_colors))

#2nd distribution of lenght of spam / not spam emails
fig2, ax = plt.subplots()
data[data['spam'] == 0]['length'].plot.hist(bins=50, alpha=0.5, color='blue',density=True, label='spam = 0', ax=ax)
data[data['spam'] == 1]['length'].plot.hist(bins=50, alpha=0.5, color='orange',density=True, label='spam = 1', ax=ax)
ax.set_xlabel('Length')
ax.set_ylabel('Frequency')
#ax.set_title('Distribution of Email Lengths')
ax.legend()

#3rd confusion matrix
#fig3, ax2 = plt.subplots()
#sns.heatmap(cm, annot=True, fmt="d")#, cmap='Blues', ax=ax2)#
#ax2.set_title('Confusion Matrix')
#ax2.set_xlabel('Predicted')
#ax2.set_ylabel('Real')

#---------------------------------------------
# Create a heatmap using Plotly
fig3, ax2 = plt.subplots()
heatmap = go.Heatmap(z=cm,
		     x=['Predicted not spam', 'Predicted spam'],
		     y=['Real not spam', 'Real spam'],
		     colorscale='Blues')
ax2 = go.Layout(title='Confusion Matrix')
fig3 = go.Figure(data=[heatmap], layout=ax2)

#distribution of email len (spam/not spam) emails


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
			st.write("Distribution of spam/not spam emails")
			st.plotly_chart(fig1,use_container_width=True) #plotly chart distribution
			st.write("spam emails represent 23.9% of total emails")
		with col2:
			st.write("Distribution spam/not spam emails lenght")
			st.pyplot(fig2) #matplotlib chart len distribution
			st.write("spam emails length distribution show that are shorter vs not spam emails")
	
	elif choice == "Predictive Model":
		st.subheader("Predictive Model")
		st.markdown("""---""")

		st.image('media/ML_spam.png')
		st.write('''Therefore, it is important to develop effective methods to classify and filter spam emails from legitimate ones. Email 
   			spam classification is the process of using machine learning algorithms to automatically identify and label spam emails 
      			based on their content, sender, subject, and other features.''')
		st.write('''Email spam classification can help users and organizations 
	 		to protect themselves from the threats of spam and improve their email experience.''')
		st.subheader("Naive Bayes")
		st.markdown("""---""")
		

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
		st.subheader("Test a new email and see if it is a spam or not")
		with st.form(key='test_email'):
			msg =st.text_area("write your email here")
			submit_code = st.form_submit_button("Execute")
		if submit_code:
			st.info("Query Result")
			if model.predict(vectorizer.transform([msg]))[0]==1:#predict_category(msg)[0]==1:
				st.write('Your message is a spam')
			else:
				st.write('Your message is a normal email')

if __name__ == '__main__':
	main()
