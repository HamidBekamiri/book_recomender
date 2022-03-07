import streamlit as st
import pandas as pd
import pickle
import time

import turicreate as tc
from streamlit import caching


st.set_page_config(page_title="Book Recommender System", page_icon="🐞", layout="centered")
st.header("🐞 Book Recommender System!")
st.subheader('David Project')

@st.cache 
def loaddata():
    df = pd.read_csv('dfrecom.csv')
    df["ProductId"] = df["ProductId"].astype(str)
    #Creating dataframe to make dictionary of books IDs and book titles 
    dfdict = df.iloc[:,[4,5,6]].copy()
    dfdict = dfdict.drop_duplicates()
    dfdict.index = range(len(dfdict))
    #These lists are for the dropbown menus
    listofbooks = list(dfdict["Book-Title"].unique())
    listofauthors = list(dfdict["Book-Author"].unique())
    return df, dfdict, listofbooks, listofauthors

model = tc.load_model("RS.model")

df, dfdict, listofbooks, listofauthors = loaddata()
IDtoNameDict = dict(zip(list(dfdict.ProductId),list(dfdict["Book-Title"])))

col_one_list_tit = listofbooks
col_one_list_auth = listofauthors

count = 1
cols_1 = st.columns((2))
selectbox_title_1 = cols_1[0].selectbox('Please choose the book title', col_one_list_tit, key = count)
selectbox_auther_1 = cols_1[1].selectbox('Please choose the author', col_one_list_auth, key = count)

count += 1
cols_2 = st.columns((2))
selectbox_title_2 = cols_2[0].selectbox('Please choose the book title', col_one_list_tit, key = count)
selectbox_auther_2 = cols_2[1].selectbox('Please choose the author', col_one_list_auth, key = count)

count += 1
cols_3 = st.columns((2))
selectbox_title_3 = cols_3[0].selectbox('Please choose the book title', col_one_list_tit, key = count)
selectbox_auther_3 = cols_3[1].selectbox('Please choose the author', col_one_list_auth, key = count)

count += 1
cols_4 = st.columns((2))
selectbox_title_4 = cols_4[0].selectbox('Please choose the book title', col_one_list_tit, key = count)
selectbox_auther_4 = cols_4[1].selectbox('Please choose the author', col_one_list_auth, key = count)

count += 1
cols_5 = st.columns((2))
selectbox_title_5 = cols_5[0].selectbox('Please choose the book title', col_one_list_tit, key = count)
selectbox_auther_5 = cols_5[1].selectbox('Please choose the author', col_one_list_auth, key = count)

cols = st.columns((1, 1))
books = []
book1 = list(dfdict[(dfdict["Book-Author"]==selectbox_auther_1)&(dfdict["Book-Title"] ==selectbox_title_1)].iloc[:,2])
book2 = list(dfdict[(dfdict["Book-Author"]==selectbox_auther_2)&(dfdict["Book-Title"] ==selectbox_title_2)].iloc[:,2])
book3 = list(dfdict[(dfdict["Book-Author"]==selectbox_auther_3)&(dfdict["Book-Title"] ==selectbox_title_3)].iloc[:,2])
book4 = list(dfdict[(dfdict["Book-Author"]==selectbox_auther_4)&(dfdict["Book-Title"] ==selectbox_title_4)].iloc[:,2])
book5 = list(dfdict[(dfdict["Book-Author"]==selectbox_auther_5)&(dfdict["Book-Title"] ==selectbox_title_5)].iloc[:,2])
books.extend(book1)
books.extend(book2)
books.extend(book3)
books.extend(book4)
books.extend(book5)
st.write(len(books))

if cols[0].button("Submit"):
    if len(books) != 0:
        books = []
        book1 = list(dfdict[(dfdict["Book-Author"]==selectbox_auther_1)&(dfdict["Book-Title"] ==selectbox_title_1)].iloc[:,2])[0]
        book2 = list(dfdict[(dfdict["Book-Author"]==selectbox_auther_2)&(dfdict["Book-Title"] ==selectbox_title_2)].iloc[:,2])[0]
        book3 = list(dfdict[(dfdict["Book-Author"]==selectbox_auther_3)&(dfdict["Book-Title"] ==selectbox_title_3)].iloc[:,2])[0]
        book4 = list(dfdict[(dfdict["Book-Author"]==selectbox_auther_4)&(dfdict["Book-Title"] ==selectbox_title_4)].iloc[:,2])[0]
        book5 = list(dfdict[(dfdict["Book-Author"]==selectbox_auther_5)&(dfdict["Book-Title"] ==selectbox_title_5)].iloc[:,2])[0]
        books.extend(book1)
        books.extend(book2)
        books.extend(book3)
        books.extend(book4)
        books.extend(book5)
 
        #passing the book IDs to one list 
        listofproducts = books
        #Making recommendation for books according to cosine similarity, passing the listofproducts to reommend
        recommendation_item = model.get_similar_items(items=listofproducts, k=10)
        #Creating dataframe
        dfitem = pd.DataFrame(recommendation_item)
        #Data manipulation and transformation to show the top 10 books to recommend
        st.table(dfitem)
        dfitem['item_occ'] = dfitem.groupby('similar').similar.transform('count')
        dfitem = dfitem.sort_values(["item_occ", "score"],ascending=(False,False))
        dfitem = dfitem[~dfitem["similar"].isin(listofproducts)]
        dfitem = dfitem.drop_duplicates(subset=['similar', "item_occ"])
        dfitem.index = range(len(dfitem))
        dfitem = dfitem.drop(columns=["ProductId", "score", "rank", "item_occ"])
        dfitem = dfitem.replace({"similar":IDtoNameDict})
        dfitem = dfitem.rename(columns={"similar":"recommended books"})
        st.write("These are the books you might be interested in, based on your previously liked books:")
        st.table(dfitem.head())
    else:
        st.write("There are no books satisfying your search!")
 



        
