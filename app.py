import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="RNN Name Generator", page_icon="🤖", layout="centered")
st.title("🤖 Pure Math RNN Name Generator")
st.write("This ultra-fast deployment loads raw neural network weight matrices directly into memory using NumPy.")

@st.cache_resource
def load_pure_math_assets():
    with open('char_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    with open('model_weights.pkl', 'rb') as f:
        weights = pickle.load(f)
    return mappings, weights

try:
    mappings, w = load_pure_math_assets()
    char_to_idx = mappings['char_to_idx']
    idx_to_char = mappings['idx_to_char']
    max_sequence_len = mappings['max_len']
    vocab_size = mappings['vocab_size']
    
    st.sidebar.header("Model Configuration")
    seed_char = st.sidebar.text_input("Starting Character(s):", value="a").lower().strip()
    temperature = st.sidebar.slider("Creativity (Temperature):", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
    num_names = st.sidebar.number_input("Count to Generate:", min_value=1, max_value=10, value=3)

    # Manual Sigmoid and Tanh activations for GRU gating math
    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    def tanh(x): return np.tanh(x)
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    # Pure NumPy Forward Pass through the GRU Network Architecture
    def predict_numpy(token_sequence):
        # 1. Embedding Layer Lookup
        seq_embed = w['embed'][token_sequence] # Shape: (seq_len, embedding_dim)
        
        # 2. Manual GRU Step Processing Loop
        h = np.zeros(128) # Initialize hidden state array vectors
        W_z, W_r, W_h = np.split(w['gru_W'], 3, axis=1)
        U_z, U_r, U_h = np.split(w['gru_U'], 3, axis=1)
        b_z, b_r, b_h = np.split(w['gru_b'], 3, axis=0)
        
        for xt in seq_embed:
            # Update gate calculation
            z = sigmoid(np.dot(xt, W_z) + np.dot(h, U_z) + b_z)
            # Reset gate calculation
            r = sigmoid(np.dot(xt, W_r) + np.dot(h, U_r) + b_r)
            # Candidate hidden state calculation
            h_tilde = tanh(np.dot(xt, W_h) + np.dot(r * h, U_h) + b_h)
            # Compute new hidden state vector
            h = (1 - z) * h + z * h_tilde
            
        # 3. Dense Softmax Output Layer Projection
        logits = np.dot(h, w['dense_W']) + w['dense_b']
        return softmax(logits)

    def sample_next_char(preds, temp):
        preds = np.log(preds + 1e-7) / temp
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)

    if st.button("Generate Names ✨"):
        if not seed_char or not all(c in char_to_idx for c in seed_char):
            st.error("Please provide valid starting letters that exist in the training vocabulary!")
        else:
            st.write("---")
            for _ in range(num_names):
                generated_name = seed_char
                for _ in range(max_sequence_len):
                    # Pad sequence manually
                    tokens = [char_to_idx[c] for c in generated_name]
                    if len(tokens) < max_sequence_len:
                        tokens = [0] * (max_sequence_len - len(tokens)) + tokens
                    else:
                        tokens = tokens[-max_sequence_len:]
                        
                    predictions = predict_numpy(np.array(tokens))
                    next_idx = sample_next_char(predictions, temperature)
                    next_char = idx_to_char[next_idx]
                    if next_char == '.':
                        break
                    generated_name += next_char
                st.success(f"💡 Generated Variant: **{generated_name.capitalize()}**")
except Exception as err:
    st.error(f"Missing Files: Please upload 'model_weights.pkl' and 'char_mappings.pkl' into GitHub. Error details: {err}")
