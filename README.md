# News_Article_Categorization_using_NLP

This project is an AI-based News Article Categorization System designed to automatically classify news articles using Natural Language Processing (NLP) and Machine Learning. It allows users to upload news articles in different file formats and analyzes the content to predict its category.

The system extracts text from the uploaded file, performs NLP preprocessing, converts the text into numerical features using TF-IDF, and uses a trained Logistic Regression model to classify the article into different news categories.

# How to Run the Project
### 📋 Prerequisites: 

     1. Python 3.x
     
     2. VS Code
     
     3. A web browser such as Google Chrome / Edge / Firefox
     
     4. Required Python libraries: Flask
                                   Pandas
                                   NumPy
                                   Scikit-learn
                                   NLTK
                                   Joblib
                                   PyPDF
                                   Python-docx
                                   OpenPyXL

##
### 🚀 Steps to Run:
#### 1️. Download / Create Project Folder

 Create or extract your project folder:

    NewsLens/

#### 2️. Open in VS Code

    File → Open Folder → NewsLens

#### 3️. Install Required Libraries

 Open the VS Code terminal and run:

    pip install -r requirements.txt

#### 4️. Run the Application

    python app.py

#### 5️. Open in Browser

 Open:

    http://127.0.0.1:5000
##
### 📝 Features
#### 1. 📂 Multi-Format File Upload

Users can upload news articles in different formats:
    
    I. PDF
    II. TXT
    III. CSV
    IV. DOCX
    V. XLSX
    VI. JSON
    VII. ZIP

The system automatically extracts readable text from the uploaded file.
##
#### 2. 🧹 NLP Text Preprocessing

The extracted article is cleaned before classification:

    I. Text Cleaning
    
    II. Tokenization
    
    III. Stop-word Removal
    
    IV. Lemmatization

    V. Removal of unnecessary characters
##    
#### 3. 📊 TF-IDF Feature Extraction

The cleaned text is converted into numerical features using:

    TF-IDF
    (Term Frequency – Inverse Document Frequency)

This helps the machine-learning model identify important words and terms in the article.
##
#### 4. 🤖 News Classification

The trained Logistic Regression model classifies the article into:

    I. ⚽ Sport
    
    II. 💼 Business
    
    III. 🏛 Politics
    
    IV. 💻 Technology
    
    V. 🎬 Entertainment
##   
#### 5. 📈 Prediction Results

After analyzing the article, the system displays:

    I. Predicted Category
    
    II. Confidence Score
    
    III. Probability of Each Category
    
    IV. Important TF-IDF Keywords
    
    V. Number of Words
    
    VI. Reading Time
    
    VII. Extracted Text Preview
##    
#### 6. 🖥️ User-Friendly Interface

The web interface provides:

    I. Drag and Drop File Upload
    
    II. File Type Display
    
    III. Analyze File Option
    
    IV. Prediction Dashboard
    
    V. Probability Visualization
    
    VI. Extracted Text Preview
##    
### ⚙️ Technologies Used

#### 1. Programming Language:

    1️. Python
    
#### 2. Frontend:
    
    1️. HTML5
    2️. CSS3
    3️. JavaScript
    
#### 3. Backend:
    
    1️. Flask

#### 4. NLP:
    
    1️. NLTK
    2️. Tokenization
    3️. Stop-word Removal
    4️. Lemmatization

#### 5. Machine Learning:

    1️. Scikit-learn
    2️. Logistic Regression
    3️. TF-IDF Vectorization

#### 6. File Processing:

    1️. PyPDF
    2️. Python-docx
    3️. OpenPyXL
    4️. Python ZIP Library

#### 7. Dataset:

    1️. BBC News Dataset

#### 8. Development Environment:

    1️. VS Code
    2️. Python 3.x
    3️. Web Browser

###
