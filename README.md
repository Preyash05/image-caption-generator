# 🖼️ AI Image Caption Generator

A Deep Learning application that generates descriptive captions for images using **DenseNet201** (for feature extraction) and **LSTM** (for text generation).



## 🛠️ Tech Stack
* **Python**
* **TensorFlow/Keras**
* **Streamlit**
* **Flickr8k Dataset**

## 📂 Project Structure
* `app.py`: The Streamlit web application.
* `image_captioning.ipynb`: The Jupyter Notebook used for training.
* `tokenizer.pkl`: The token dictionary for text processing.

## 🔧 How to Run Locally
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  **Note:** You must generate or download the `model.keras` file (it is too large for GitHub) and place it in the root folder.
4.  Run the app:
    ```bash
    streamlit run app.py
    ```