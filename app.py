import pandas as pd
import streamlit as st
import pickle as pkl

# -----------------------------
# Load Model
# -----------------------------
with open("model/model.pkl", "rb") as file:
    model = pkl.load(file)


# -----------------------------
# Load Encoder
# -----------------------------
with open("model/encoder.pkl", "rb") as file:
    encoder = pkl.load(file)

# -----------------------------
# Streamlit UI
# -----------------------------


st.set_page_config(page_title="Flight Ticket Price Predictor",
                   page_icon="✈️")
st.title("✈️Flight Ticket Price Predictor")


cities = ['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi']
airlines = {'SpiceJet':'SpiceJet', 'AirAsia':'AirAsia',
            'Indigo':'Indigo', 'Air India':'Air_India'}

st.write("")
# Taking Inputs 

from_city = st.selectbox("From : ",cities)

to_city = [city for city in cities if city != from_city]
to_city = st.selectbox("To : ",to_city)

airline = st.selectbox("Airline : ",airlines)

fly_class = st.selectbox("Class : ",['Economy', 'Business'])

duration = st.number_input("Duration (hours): ",min_value=1.0,max_value=5.0,step=0.1, value=3.0)



# ---------------------------------
# New Flight Data and Predict Price
# ---------------------------------
if st.button("Predict Price"):
    new_flight = pd.DataFrame([
        {
            "from": from_city,
            "to": to_city,
            "airline": airlines[str(airline)],
            "class": fly_class,
            "duration": duration
        }])
    encoded = encoder.transform(
        new_flight[["from", "to", "airline", "class"]]
        )
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            ["from", "to", "airline", "class"]
            ),index=new_flight.index)

    X_new = pd.concat([
        encoded_df,
        new_flight[["duration"]]
        ],axis=1)
    X_new.columns = X_new.columns.astype(str)
    predicted_price = model.predict(X_new)

    
    # -----------------------------
    # Display Result
    # -----------------------------
    st.success(
        f"""
        Airlines : {airline}
        
        From : {from_city}

        To : {to_city}

        Class : {fly_class}

        Estimated Price : ₹ {predicted_price[0]:.2f}
       """)

