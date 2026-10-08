import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


st.title("Streamlit Iris Prediction")
st.write("A simple model")

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x_train,  y_train)
accuracy = model.score(x_test, y_test)

st.write(f"Model Accuracy:{accuracy:.0%}")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0 )
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0 )
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0 )
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0 )


features = [[sepal_length, sepal_width, petal_length, petal_width]]

if st.button("Predict"):
    predict = model.predict(features)[0]
    st.success(f"Prediction: {iris.target_names[predict]}")

    st.write(predict)



