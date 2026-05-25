import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

st.set_page_config(page_title="RNN Name Generator", page_icon="🤖", layout="centered")

st.title("🤖 Character-Level RNN Name Generator")
st.write("This Deep Learning model utilizes a Recurrent Neural Network (GRU) to generate entirely original names character-by-character.")

# Safely cache model assets so they don't reload on every button click
@st.cache_resource
def load_assets():
    # Streamlit looks directly in your root repository folder for these files
    model = tf.keras.models.load_model('name_generator_model.keras')
    with open('char_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    return model, mappings

try:
    model, mappings = load_assets()
    char_to_idx = mappings['char_to_idx']
    idx_to_char = mappings['idx_to_char']
    max_sequence_len = mappings['max_len']
    vocab_size = mappings['vocab_size']
    
    # UI Layout controls in the Sidebar
    st.sidebar.header("Model Configuration")
    seed_char = st.sidebar.text_input("Starting Character(s):", value="a").lower().strip()
    temperature = st.sidebar.slider("Creativity (Temperature):", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
    num_names = st.sidebar.number_input("Count to Generate:", min_value=1, max_value=10, value=3)

    # Temperature Sampling Logic to avoid repetitive prediction patterns
    def sample_next_char(preds, temp):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-7) / temp
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    if st.button("Generate Names ✨"):
        if not seed_char or not all(c in char_to_idx for c in seed_char):
            st.error("Please provide valid starting letters that exist in the training vocabulary!")
        else:
            st.write("---")
            for _ in range(num_names):
                generated_name = seed_char
                
                for _ in range(max_sequence_len):
                    # Format current sequence string for evaluation
                    tokens = [char_to_idx[c] for c in generated_name]
                    tokens = tf.keras.preprocessing.sequence.pad_sequences(
                        [tokens], maxlen=max_sequence_len, padding='pre'
                    )
                    
                    # Predict distribution map
                    predictions = model.predict(tokens, verbose=0)[0]
                    next_idx = sample_next_char(predictions, temperature)
                    next_char = idx_to_char[next_idx]
                    
                    if next_char == '.':
                        break
                    generated_name += next_char
                
                st.success(f"💡 Generated Variant: **{generated_name.capitalize()}**")

except Exception as err:
    st.error(f"Error initializing app configuration assets: {err}. Please make sure your model files are uploaded in the same repository folder.")
