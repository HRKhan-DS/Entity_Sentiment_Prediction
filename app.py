import streamlit as st
import pickle
import os
import re

try:
    import spacy
    from spacy.cli.download import download as spacy_download
    from spacy import displacy
    from spacy.lang.en.stop_words import STOP_WORDS
except ImportError as e:
    st.error(f"Error importing SpaCy: {e}. Please ensure SpaCy is correctly installed.")

from sentence_transformers import SentenceTransformer
import string
from PIL import Image
import spacy_streamlit
from streamlit_option_menu import option_menu

# Page setup
st.set_page_config(page_title="Entity-Sentiment Analysis",
                   layout='wide',
                   page_icon="🤗")

# Cache the function to prevent re-downloading the model
@st.cache_resource
def load_model():
    # Attempt to load the model
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        # If the model is not found, download and then load it
        spacy_download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")
    return nlp

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
        st.image(img)

    if selected == 'Description':
        st.subheader("Description")
        st.write("This application provides entity-level sentiment analysis using a machine learning model.")
        st.write("Entity-level sentiment analysis involves detecting entities mentioned in a text and determining the sentiment expressed towards those entities.")
        st.write("It can be applied in various domains such as product reviews, social media monitoring, and customer feedback analysis to understand the sentiments associated with specific entities (e.g., brands, products, people).")
        st.write("By identifying and analyzing sentiments at the entity level, businesses can gain deeper insights into customer opinions and make informed decisions.")
        
    elif selected == 'Entity Recognition':
        text_input = st.text_area("Enter text for Entity Recognition", "")

        try:
            nlp = load_model()

            if text_input:
                doc = nlp(text_input)
                # Use displacy to visualize entity recognition with reduced gap
                html = displacy.render(doc, style="ent", page=True, options={"distance": 90})  # Adjust distance parameter
                st.components.v1.html(html, height=200)
            
            else:
                st.write("Please enter some text for Entity Recognition.")

            # Add the Submit button
            if st.button("Submit"):
                pass

        except Exception as e:
            st.error(f"Error loading the model: {e}")

    elif selected == "Sentiment Analysis":
        text_input = st.text_area("Enter text for sentiment analysis", "")

        try:
            nlp = load_model()
        
            rf_model_path = os.path.join(os.path.dirname(__file__), 'model/random_forest_model.sav')
            sn_model_path = os.path.join(os.path.dirname(__file__), 'model/sentence_trans_model.sav')
            
            rf_model = pickle.load(open(rf_model_path, 'rb'))
            sn_model = SentenceTransformer(sn_model_path)
        except Exception as e:
            st.error(f"Error loading the sentiment analysis models: {e}")
            return
        
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
                st.write("Please enter some text for sentiment analysis.")

if __name__ == "__main__":
    main()

