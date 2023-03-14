# By David Maimoun
# deployed the 13.03.23
import streamlit as st
import pandas as pd
import json
from streamlit_lottie import st_lottie
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

st.write("""
<style>
    h1 {
    color: dodgerblue;
    }
    .result {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
</style>
""", unsafe_allow_html=True)

def load_lottiefile(filepath: str):
   with open(filepath, "r") as f:
      return json.load(f)

def returnLottie(path):
   return st_lottie(
      load_lottiefile(path),
      speed=1,
      reverse=False,
      loop=True,
      quality="high", # medium ; high
      # renderer="svg", # canvas
      height=None,
      width=None,
      key=None,
   )


st.markdown("<h1>DiabetesPredictor App</h1><br>",unsafe_allow_html=True)
        
st.sidebar.header("DiabetesPredictor")
st.sidebar.markdown("<br>",unsafe_allow_html=True)
def user_input_features():
   age = st.sidebar.slider('1- Age', 0, 120, 31)
   pregnancies = st.sidebar.slider('2- Pregnancies', 0, 30, 1)
   glucose = st.sidebar.number_input('3- Glucose', value = 85)
   bloodPressure = st.sidebar.number_input('4- Blood Pressure', value = 66)
   skinThickness = st.sidebar.number_input('5- Skin Thickness', value = 29)
   insulin = st.sidebar.number_input('6- Insulin', value = 0)
   bmi = st.sidebar.number_input('7- BMI', value = 26.6)
   diabetesPedigreeFunction = st.sidebar.number_input('8- Diabete Pedigree', value = 0.351)

   data = {
      'Pregnancies': pregnancies, 
      'Glucose': glucose,
      'BloodPressure': bloodPressure,
      'SkinThickness': skinThickness,
      'Insulin': insulin,
      'BMI': bmi,
      'DiabetesPedigreeFunction': diabetesPedigreeFunction,
      'Age': age
   }
   features = pd.DataFrame(data, index=[0])
   return features

my_data = user_input_features()

st.write("Patient data:")
st.dataframe(my_data)

df = pd.read_csv('data/diabetes.csv')

X = df.drop(columns = 'Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
std_data = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(std_data, y, test_size = 0.2, stratify = y)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
# my_data_array = np.asarray(my_data)
# my_data_reshaped = my_data_array.reshape(1, -1)

std_data = scaler.transform(my_data)
prediction = classifier.predict(std_data)

is_has_disease = "Has" if prediction[0] == 1 else "Doesn't Have"
style = "color:red" if prediction[0] == 1 else "color:dodgerblue"

col1, col2 = st.columns([3,1])
with col1:
   st.markdown(f"""
   The patient 
   <span class='result' style={style}>{is_has_disease}</span> a Diabetes
   with a probability of <span class='result' style={style}>{round(training_data_accuracy, 2)*100}%</span>.
""", unsafe_allow_html = True)
with col2:
      returnLottie("assets/doctor2.json")

st.markdown('<br><br><p><i>By David Maimoun</p></i>',unsafe_allow_html=True)  
