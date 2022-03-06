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
    model = tc.load_model("/model/")
    df = pd.read_csv('dfrecom.csv', nrow=100)
    df["ProductId"] = df["ProductId"].astype(str)
    #Creating dataframe to make dictionary of books IDs and book titles 
    dfdict = df.iloc[:,[4,5,6]].copy()
    dfdict = dfdict.drop_duplicates()
    dfdict.index = range(len(dfdict))
    IDtoNameDict = dict(zip(list(dfdict.ProductId),list(dfdict["Book-Title"])))
    #These lists are for the dropbown menus
    listofbooks = list(dfdict["Book-Title"].unique())
    listofauthors = list(dfdict["Book-Author"].unique())
    return model, df, dfdict, IDtoNameDict, listofbooks, listofauthors

model, df, dfdict, IDtoNameDict, listofbooks, listofauthors = loaddata()
col_one_list_tit = listofbooks[:100]
col_one_list_auth = listofauthors[:100]
cols = st.columns((2))
selectbox_title = cols[0].selectbox('Please choose the book title', col_one_list_tit, index=0)
selectbox_auther = cols[1].selectbox('Please choose the author', col_one_list_auth)

book1 = list(dfdict[(dfdict["Book-Author"]==selectbox_auther)&(dfdict["Book-Title"] ==selectbox_title)].iloc[:,2])[0]
#passing the book IDs to one list 
listofproducts = [book1]
#Making recommendation for books according to cosine similarity, passing the listofproducts to reommend
recommendation_item = model.get_similar_items(items=listofproducts, k=10)
#Creating dataframe
dfitem = pd.DataFrame(recommendation_item)
#Data manipulation and transformation to show the top 10 books to recommend
dfitem['item_occ'] = dfitem.groupby('similar').similar.transform('count')
dfitem = dfitem.sort_values(["item_occ", "score"],ascending=(False,False))
dfitem = dfitem[~dfitem["similar"].isin(listofproducts)]
dfitem = dfitem.drop_duplicates(subset=['similar', "item_occ"])
dfitem.index = range(len(dfitem))
dfitem = dfitem.drop(columns=["ProductId", "score", "rank", "item_occ"])
dfitem = dfitem.replace({"similar":IDtoNameDict})
dfitem = dfitem.rename(columns={"similar":"recommended books"})


cols_2 = st.columns((1, 1))
        
if cols_2[0].button("Submit"):
    if len(dfitem) == 0:
        st.write("There are no books satisfying your search!")
    else:
        st.write("These are the books you might be interested in, based on your previously liked books:")
        st.table(dfitem.head())
