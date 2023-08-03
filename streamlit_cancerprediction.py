# Import libraries
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import pickle, joblib

# Load the saved model
model = pickle.load(open('knn.pkl', 'rb'))
ct1 = joblib.load('processed1')
ct2 = joblib.load('processed2')


def predict(data, user, pw, db):
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

    data.drop(['id'], axis = 1, inplace = True) # Excluding id column
    newprocessed1 = pd.DataFrame(ct1.transform(data), columns = data.columns)
    newprocessed2 = pd.DataFrame(ct2.transform(newprocessed1), columns = newprocessed1.columns)
    predictions = pd.DataFrame(model.predict(newprocessed2), columns = ['diagnosis'])
    
    final = pd.concat([predictions, data], axis = 1)
    final.to_sql('cancer_predictions', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final


def main():  

    st.title(" Cancer Prediction")
    st.sidebar.title("Caneer Prediction")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Cancer Prediction </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    
    uploadedFile = st.sidebar.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("You need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict(data, user, pw, db)
                                   
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm).set_precision(2))
                           
if __name__=='__main__':
    main()


