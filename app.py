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

    # Manual Activation functions
    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    def tanh(x): return np.tanh(x)
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    # Pure NumPy Forward Pass through the GRU Network Architecture
    def predict_numpy(token_sequence):
        # 1. Embedding Layer Lookup
        seq_embed = w['embed'][token_sequence]
        
        # 2. Extract and Split GRU weights cleanly
        hidden_dim = 128
        W_z, W_r, W_h = w['gru_W'][:, :hidden_dim], w['gru_W'][:, hidden_dim:hidden_dim*2], w['gru_W'][:, hidden_dim*2:]
        U_z, U_r, U_h = w['gru_U'][:, :hidden_dim], w['gru_U'][:, hidden_dim:hidden_dim*2], w['gru_U'][:, hidden_dim*2:]
        
        # Handle the Keras double-bias packing vector safely
        input_bias = w['gru_b'][0] if len(w['gru_b'].shape) > 1 else w['gru_b']
        b_z, b_r, b_h = input_bias[:hidden_dim], input_bias[hidden_dim:hidden_dim*2], input_bias[hidden_dim*2:hidden_dim*3]
        
        h = np.zeros(hidden_dim) # Initialize hidden state
        
        for xt in seq_embed:
            z = sigmoid(np.dot(xt, W_z) + np.dot(h, U_z) + b_z)
            r = sigmoid(np.dot(xt, W_r) + np.dot(h, U_r) + b_r)
            h_tilde = tanh(np.dot(xt, W_h) + np.dot(r * h, U_h) + b_h)
            h = (1 - z) * h + z * h_tilde
            
        # 3. Dense Softmax Output Output
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
    st.error(f"Error executing math layers: {err}")


       
       
                
