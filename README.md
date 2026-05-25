# 🤖 Character-Level RNN Name Generator

A Deep Learning-driven text generation application that utilizes a Recurrent Neural Network (RNN) architecture to generate completely original, realistic names character-by-character. This project demonstrates end-to-end data processing, sequential modeling, text tokenization, and cloud application deployment for natural language processing (NLP) tasks.

---

## 📊 How It Works (Methodology)

Instead of relying on a hardcoded dictionary or simple random combinations, this project leverages sequence prediction:

1. **Text Vectorization:** Raw name strings are cleaned, appended with an end-of-sequence token (`.`), and converted into numerical token matrices.
2. **Sliding Window Training:** The training step slices the name data into partial sliding sequences. For example, given the sequence input `a-m-a-n`, the network learns to target and predict `i` as the next character to complete the word `amani`.
3. **Deep Learning Network:** The model passes token integers through an **Embedding layer**, feeds them into a hidden **Gated Recurrent Unit (GRU)** layer to catch long-term character dependencies, and maps them to a **Dense layer with a Softmax activation function** to get probability distributions across the entire vocabulary.
4. **Temperature-Scaled Sampling:** A custom temperature script modifies the raw prediction probabilities during inference, allowing users to tune the balance between conservative patterns and creative variations.

---

## 🛠️ Tech Stack & Architecture

* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Dataset Management:** NumPy
* **Deployment & Frontend UI:** Streamlit Cloud
* **Serialization:** Pickle (for structural vocabulary lookup configurations)

---

## 🚀 How to Run Locally

If you wish to clone this repository and run the interface on your local machine, execute the following commands in your terminal:

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/name-generator-rnn.git](https://github.com/YOUR_USERNAME/name-generator-rnn.git)
cd name-generator-rnn

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
