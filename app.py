import streamlit as st
import pickle
import os
import re
import spacy
from sentence_transformers import SentenceTransformer
import string
from spacy.lang.en.stop_words import STOP_WORDS
from PIL import Image
import spacy_streamlit
from streamlit_option_menu import option_menu

# Page setup
st.set_page_config(page_title="Entity-Sentiment Analysis",
                   layout='wide',
                   page_icon="🤗")

def main():
    with st.sidebar:
        selected = option_menu("Entity-Senti App",
                               ["Description", "Entity Recognition", "Sentiment Analysis"],
                               icons=["info_circle", "activity", 'activity'],
                               menu_icon='cast',
                               default_index=0)
        
        img_path = os.path.join(os.path.dirname(__file__), 'static/first_01.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            st.image(img, width=290)

    st.header("Welcome to Entity Level-Sentiment Prediction App!")
    img_path = os.path.join(os.path.dirname(__file__), 'static/pic_for_display_01.jpg')
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, width=800)

    if selected == 'Description':
        st.subheader("Description")
        st.write("This application provides entity-level sentiment analysis using a machine learning model.")
        st.write("Entity-level sentiment analysis involves detecting entities mentioned in a text and determining the sentiment expressed towards those entities.")
        st.write("It can be applied in various domains such as product reviews, social media monitoring, and customer feedback analysis to understand the sentiments associated with specific entities (e.g., brands, products, people).")
        st.write("By identifying and analyzing sentiments at the entity level, businesses can gain deeper insights into customer opinions and make informed decisions.")
        
    elif selected == 'Entity Recognition':
        text_input = st.text_area("Enter text for analysis", "")
        
        import subprocess

        @st.cache_resource
        def download_en_core_web_sm():
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

        # Download the 'en_core_web_sm' model if not already downloaded
        download_en_core_web_sm()

        # Load the 'en_core_web_sm' model
        nlp = spacy.load("en_core_web_sm")
 
        doc = nlp(text_input)

        if st.button("Submit"):
            if text_input:
                entities = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
                entity_str = ", ".join([f"{ent[0]} ({ent[1]})" for ent in entities])
                st.markdown(f"**Entity Recognition:** <span style='color:blue'>{entity_str}</span>", unsafe_allow_html=True)
            else:
                st.write("Please enter some text for analysis.")
    
    elif selected == "Sentiment Analysis":
        text_input = st.text_area("Enter text for analysis", "")

        @st.cache_resource()
        def get_model():
            try:
                return spacy.load("en_core_web_sm")
            except OSError:
                spacy.cli.download("en_core_web_sm")
                return spacy.load("en_core_web_sm")

        nlp = get_model()
        
        rf_model_path = os.path.join(os.path.dirname(__file__), 'model/random_forest_model.sav')
        rf_model = pickle.load(open(rf_model_path, 'rb')) 
        
        sn_model_path = os.path.join(os.path.dirname(__file__), 'model/sentence_trans_model.sav')
        sn_model = pickle.load(open(sn_model_path, 'rb'))
        
        def clean_text(text):
            pattern = r"[^a-zA-Z]+|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
            cleaned_text = re.sub(pattern, ' ', text)
            return cleaned_text.lower()

        def spacy_token(sentence):
            doc = nlp(sentence)
            mytokens = [word.lemma_.lower().strip() for word in doc]
            mytokens = [word for word in mytokens if word not in STOP_WORDS and word not in string.punctuation]
            return " ".join(mytokens)
        
        if st.button("Submit"):
            if text_input:
                cleaned_text = clean_text(text_input)
                spacy_text = spacy_token(cleaned_text)
                embeddings = sn_model.encode([spacy_text])
                prediction = rf_model.predict(embeddings)
                
                sentiment_label = "Positive" if prediction[0] == 1 else "Negative"
                st.write(f"Sentiment: {sentiment_label}")

                img_path = os.path.join(os.path.dirname(__file__), f'static/{sentiment_label.lower()}.jpg')
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, width=100)
            else:
                st.write("Please enter some text for analysis.")

if __name__ == "__main__":
    main()



