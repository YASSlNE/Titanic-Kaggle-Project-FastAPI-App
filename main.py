from fastapi import FastAPI, Request
import pandas as pd
import numpy as np
import pickle

file1 = open('titanic_model.sav', 'rb')
lr = pickle.load(file1)
file1.close()


app = FastAPI()


@app.get("/api")
async def read_item(pclass=1, sex="Female", age=38, sibsp=1, parch=0, fare=71.2833, embarked=0):




    if(sex.upper()=="FEMALE"):
        sex=0
    elif(sex.upper()=="MALE"):
        sex=1
    else:
        return{
            "Invalid sex"
        }


    # Converting the passed args to a pandas DataFrame
    X = pd.DataFrame({"Pclass":pclass,"Sex":sex, "Age":age, "SibSp":sibsp,	"Parch":parch,	"Fare":fare		,"Embarked":embarked}, index=[0])


    prediction=lr.predict(X)




    if(prediction[0]):
        prediction="Survived"
    else:
        prediction="D34D"

    response = {
        "prediction":prediction,
    }

    return response
