import pandas as pd
from bs4 import BeautifulSoup
import re
import streamlit as st
import nltk

from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
class CleanData:
    def __init__(self,data):
        self.data = data
    def clean(self):
        data_metadata={'empty label':0,'invalid label':0,'missing label':0,'missing subject':0,'missing body':0,'removed duplicates':0}
        with st.spinner("Cleaning data..."):
            if "body" not in self.data.columns or "label" not in self.data.columns:
                st.error(f"body and label must be present in the data")
                st.stop()

            if self.data["label"].isnull().any():
                missing_count = self.data["label"].isnull().sum()
                data_metadata['missing label']=missing_count
            if "subject" in self.data.columns:
                self.data["subject"] = self.data["subject"].fillna("")
                data_metadata['missing subject']=self.data["subject"].isnull().sum()

            self.data["body"] = self.data["body"].fillna("")

            data_metadata['missing body']=self.data["body"].isnull().sum()

            cleaned_data = self.data.drop_duplicates()
            cleaned_data = cleaned_data[cleaned_data["body"].str.strip() != ""].copy()
            data_metadata['removed duplicates']=self.data.shape[0]-cleaned_data.shape[0]

            label_text = cleaned_data["label"].astype(str).str.strip()
            empty_label_mask = cleaned_data["label"].isna() | (label_text == "")
            data_metadata['empty label'] = int(empty_label_mask.sum())
            cleaned_data = cleaned_data.loc[~empty_label_mask].copy()

            numeric_labels = pd.to_numeric(cleaned_data["label"].astype(str).str.strip(), errors="coerce")
            invalid_label_mask = numeric_labels.isna()
            data_metadata['invalid label'] = int(invalid_label_mask.sum())
            if invalid_label_mask.any():
                invalid_examples = cleaned_data.loc[invalid_label_mask, "label"].astype(str).head(5).tolist()
                st.warning(
                    "Rows with labels that are not numbers were removed. "
                    
                )
                cleaned_data = cleaned_data.loc[~invalid_label_mask].copy()
                numeric_labels = numeric_labels.loc[~invalid_label_mask]

            if cleaned_data.empty:
                st.error("No rows remain after removing empty or invalid labels.")
                st.stop()

            cleaned_data["label"] = numeric_labels.astype(int)
            cleaned_data["label"] = cleaned_data["label"].apply(
                lambda x: 0 if x == 0 else 1
            )
            if "subject" in cleaned_data.columns:

                cleaned_data["email_text"] = (
                        cleaned_data["subject"] + " " + cleaned_data["body"]
                )
            else:
                cleaned_data["email_text"] = cleaned_data["body"]

            cleaned_data["email_text"] = cleaned_data["email_text"].apply(self.remove_html)
            cleaned_data["email_text"] = cleaned_data["email_text"].str.lower()

            cleaned_data["email_text"] = cleaned_data["email_text"].apply(
                lambda x: re.sub(r"http\S+|www\S+", "", x)
            )




            cleaned_data["email_text"] = cleaned_data["email_text"].apply(
                lambda x: re.sub(r"\S+@\S+", "", x)
            )


            cleaned_data["email_text"] = cleaned_data["email_text"].str.replace(r"[^\w\s]", "", regex=True)

            cleaned_data["email_text"] = cleaned_data["email_text"].str.replace(r"\d+", "", regex=True)

            cleaned_data["email_text"] = cleaned_data["email_text"].str.replace(r"\s+", " ", regex=True).str.strip()

            cleaned_data["email_text"] = cleaned_data["email_text"].fillna("").astype(str)
            st.empty().info(
                f"Empty labels: {data_metadata['empty label']}, "
                f"invalid labels: {data_metadata['invalid label']}, "
                f"missing labels: {data_metadata['missing label']}, "
                f"missing subjects: {data_metadata['missing subject']}, "
                f"missing bodies: {data_metadata['missing body']}, "
                f"removed duplicates: {data_metadata['removed duplicates']}."
            )
        with st.spinner("tokenizing and lemmatizing..."):
            cleaned_data["tokens"] = cleaned_data["email_text"].apply(wordpunct_tokenize)

            stop_words = set(stopwords.words("english"))

            cleaned_data["tokens"] = cleaned_data["tokens"].apply(
                lambda words: [word for word in words if word not in stop_words]
            )

            lemmatizer = WordNetLemmatizer()

            cleaned_data["tokens"] = cleaned_data["tokens"].apply(
                lambda words: [lemmatizer.lemmatize(word) for word in words]
            )
            cleaned_data["clean_email"] = cleaned_data["tokens"].apply(
                lambda words: " ".join(words)
            )
            with st.expander("Cleaned Data Preview"):
                st.dataframe(cleaned_data.head(), width="stretch")
                st.write("Class distribution")
                st.write(cleaned_data["label"].value_counts())
        return {'data':cleaned_data,'metadata':data_metadata,'cleaned':True,'label_count':cleaned_data["label"].value_counts()}

    @staticmethod
    def remove_html(text):
        if pd.isna(text):
            return ""
        text = str(text)
        if not re.search(r"<[a-zA-Z][^>]*>", text):
            return text
        return BeautifulSoup(text, "html.parser").get_text()
