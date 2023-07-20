import streamlit as st
import pandas as pd
import numpy as np

st.write('I hate windows')
st.write(pd.DataFrame({
  'first_column':[1,2,3,4],
  'second_column':[10,20,30,40]
}))

data=pd.DataFrame({
  'first_column':[5,6,7,8],
  'second_column':[15,25,35,45]
})

st.write(data)

st.markdown('''# Markdown
# For Heading Level -1 or Title use (#)
## For Heading Level -2 or Header use (##)
### For Heading Level -3 or Subheader use (###)
#### For Heading Level -4 
---
To create paragraphs, use a blank line to separate one or more lines of text. 

Emphasis

#### Making text bold, italic, bold italic
To **bold text**, add two asterisks or underscores before and after a word or phrase.

To *italicize text*, add one asterisk or underscore before and after a word or phrase.

Add three asterisks or underscores before and after a word or ***phrase for bold italic***
---
#### Blockquotes

To create a blockquote, add a > in front of a paragraph.
>Dorothy followed her through many of the beautiful rooms in her castle.
>>The Witch bade her clean the pots and kettles and sweep the floor and keep the fire fed with wood.
---
#### List

Order List (items with number)

1.First item
2.Second item
3.Third item
4.Fourth item

  1.Intendent Item1
  2.Intendent Item2
---
#### Unordered List
add dashes (-), asterisks (*), or plus signs (+) in front of line items.
- First item
- Second item
- Third item
- Fourth item
---
#### Links

To create a link, enclose the link text in brackets and then follow it immediately with the URL in parentheses

My favorite search engine is [Google](https://google.com).

---
#### URLs and Email Address

To quickly turn a URL or email address into a link, enclose it in angle brackets.
<https://www.ie.edu/school-science-technology/programs/master-business-analytics-big-data/>
<concepciond@faculty.ie.edu>

---
#### Images

To add an image, add an exclamation mark (!), followed by alt text in brackets, and the path or URL to the image asset in parentheses. You can optionally add a title in quotation marks after the path or URL. ![Mountains are beautiful](https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80)''')

st.markdown('---')

st.title('title')

st.header('header')

st.subheader('header')

body='''
  import streamlit as st
  st.write('I love streamlit')
  st.write(123456)
  '''
st.code(body,language='python')

formula='''
  (x^2+y^2=z^2)
  '''
st.latex(formula)


data=pd.read_csv('tips.csv')
st.dataframe(data)

st.table(data)


json_data=data.head(3).to_dict()
st.json(json_data)

st.markdown('---')

st.info('this is an info message')
st.error('ups this is an error')
st.warning('take care')
st.success('yeeeah you get it')

import time
time.sleep(5)
st.balloons()


st.markdown('---')
st.header('this is super cool:hotdog:')

st.markdown('---')
st.image('media/fotito.jpeg')

st.markdown('---')

#video_file=open('media/google_team.mp4','rb')
#video_bytes=video_file.read()
#st.video(video_bytes)

st.markdown('---')
audio_file=open('media/audio.mp3','rb')
audio_bytes=audio_file.read()
st.audio(audio_bytes)

st.markdown('---')

side_bar=st.sidebar
side_bar.write('this is on the left')
side_bar.write('this is another')
st.write('this is in the middle')

st.markdown('---')
c1,c2,c3=st.columns(3)
with c1:
  st.image('media/column1.jpg')
with c2:
  st.image('media/column2.jpg')
with c3:
  st.image('media/column3.jpg')
