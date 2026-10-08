from flask import Flask,render_template,request
import joblib 
import pandas as pd 

app= Flask(__name__)

# Load trained model 
model = joblib.load("models\car_price_pickle.pkl")

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    present_price=float(request.form["present_price"])
    kms_driven= float(request.form["kms_driven"])
    fuel_type = request.form["fuel_type"]
    seller_type= request.form["seller_type"]
    transmission= request.form["transmission"]
    owner= int(request.form['owner'])
    car_age= int(request.form["car_age"])

    new_car =pd.DataFrame({
        "Present_Price" : [present_price],
        "Kms_Driven" : [kms_driven],
        "Fuel_Type" : [fuel_type],
        "Seller_Type" : [seller_type],
        "Transmission" : [transmission],
        "Owner" : [owner],
        "Car_Age" : [car_age]
    })
    prediction = model.predict(new_car)
    predicted_price = round(prediction[0],2)
    return render_template("index.html",prediction=predicted_price)
if __name__ == "__main__":
    app.run(debug=True)