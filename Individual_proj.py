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
	#vectorizing 
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

#calculate accuracy of the model
accuracy=100 * sum(predictions == Y_test) / len(predictions)
cm = confusion_matrix(Y_test, predictions)

#function predict spam/not spam emails
@st.cache_data 
def predict_category(s, model=model):
    pred = model.predict(vectorizer.transform([s]))
    return pred
	
#---------------------------
#------Create charts--------
#---------------------------
#1st pie chart with distribution
df2=data.groupby('spam').count().reset_index().replace(0,"not spam").replace(1,"spam")
custom_colors = ['blue', 'orange'] 
fig1 = px.pie(df2,
             values='text',
             names='spam',
             title='Distribution of spam/not spam emails',
             labels={'text':'# of cases'})
fig1.update_traces(marker=dict(colors=custom_colors))

#2nd distribution of lenght of spam / not spam emails
fig2, ax = plt.subplots()
# Plot the histogram for spam = 0 in blue
data[data['spam'] == 0]['length'].plot.hist(bins=50, alpha=0.5, color='blue',density=True, label='spam = 0', ax=ax)
# Plot the histogram for spam = 1 in orange
data[data['spam'] == 1]['length'].plot.hist(bins=50, alpha=0.5, color='orange',density=True, label='spam = 1', ax=ax)
# Add labels and legend
ax.set_xlabel('Length')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Email Lengths')
ax.legend()

#3rd confusion matrix
fig3, ax = plt.subplots()
sns.heatmap(cm, annot=True)


#fig3=px.imshow(cm)
@st.cache_data 
def plot_matrix(cm, classes):
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    return cm_df

	#distribution of email len (spam/not spam) emails


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
		st.plotly_chart(fig1) #plotly chart
		st.markdown("""---""")
		st.pyplot(fig2) #matplotlib chart

	elif choice == "Predictive Model":
		st.subheader("Predictive Model")
		st.write('I love windows')

	elif choice == "Results":
		st.subheader("Results")
		st.markdown("""---""")
		st.write("Confusion Matrix:")
		#st.dataframe(plot_matrix(cm, ['not spam', 'spam']))
		st.pyplot(fig3)
		st.write(f"Accuracy: {accuracy:.2f}%")
		
		st.markdown("""---""")
		st.subheader("Test a new email and see if it is a spam or not")
		with st.form(key='test_email'):
			msg =st.text_area("write your email here")
			submit_code = st.form_submit_button("Execute")
		if submit_code:
			st.info("Query Result")
			if predict_category(msg)[0]==1:
				st.write('Your message is a spam')
			else:
				st.write('Your message is a normal email')
						
	else:	
		st.subheader("About Me")
		st.write('yo amo windows')





if __name__ == '__main__':
	main()
