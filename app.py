
from flask import Flask, render_template, request
import numpy as np
import pickle as pkl

app = Flask(__name__)


# ==========================================
# 1. LOAD TRAINED MODELS AND ENCODER
# ==========================================

with open('dtc_model.pkl', 'rb') as f:
    dtc = pkl.load(f)

with open('dtr_model.pkl', 'rb') as f:
    dtr = pkl.load(f)

with open('ohe (2).pkl', 'rb') as f:
    ohe = pkl.load(f)


"""
0   origin_hub                        object
1   destination_spot                  object
2   district                          object
3   vehicle_or_ride_type              object
4   season_name                       object
5   group_size                        int64
6   units_hired                       int64
7   trip_duration_hours               float64
8   extra_waiting_hours               float64
9   route_distance_km                 float64
10  terrain_difficulty                object
11  outside_cabs_restricted           int64
12  booking_channel                   object
13  quoted_total_fare_inr             int64
14  quoted_per_person_inr             int64
15  official_fair_price_inr           int64
16  max_reasonable_seasonal_fare_inr  int64
17  quote_fairness_label              object
"""


# ==========================================
# 2. HOME ROUTE
# ==========================================

@app.route('/', methods=['GET', 'POST'])
def index():

    prediction = None
    error_message = ""

    if request.method == 'POST':

        try:

            # ==================================
            # NUMERICAL FEATURES
            # ==================================

            group_size = int(request.form['group_size'])

            units_hired = int(request.form['units_hired'])

            trip_duration_hours = float(
                request.form['trip_duration_hours']
            )

            extra_waiting_hours = float(
                request.form['extra_waiting_hours']
            )

            route_distance_km = float(
                request.form['route_distance_km']
            )

            outside_cabs_restricted = int(
                request.form['outside_cabs_restricted']
            )

            quoted_total_fare_inr = float(
                request.form['quoted_total_fare_inr']
            )

            if group_size <= 0:
                raise ValueError("Group size must be greater than zero")

            if units_hired <= 0:
                raise ValueError("Units hired must be greater than zero")

            quoted_per_person_inr = round(
                quoted_total_fare_inr / group_size
            )


            # ==================================
            # CATEGORICAL FEATURES
            # ==================================

            origin_hub = request.form['origin_hub']

            destination_spot = request.form['destination_spot']

            district = request.form['district']

            vehicle_or_ride_type = request.form['vehicle_or_ride_type']

            season_name = request.form['season_name']

            terrain_difficulty = request.form['terrain_difficulty']

            booking_channel = request.form['booking_channel']


            # ==================================
            # STORE ALL FEATURES IN ONE LIST
            # ==================================

            cab_booking_details = [[

                origin_hub,
                destination_spot,
                district,
                vehicle_or_ride_type,
                season_name,
                group_size,
                units_hired,
                trip_duration_hours,
                extra_waiting_hours,
                route_distance_km,
                terrain_difficulty,
                outside_cabs_restricted,
                booking_channel,
                quoted_total_fare_inr,
                quoted_per_person_inr

            ]]


            # ==================================
            # NUMERICAL FEATURES
            # ==================================

            x_n = [[

                cab_booking_details[0][5],
                cab_booking_details[0][6],
                cab_booking_details[0][7],
                cab_booking_details[0][8],
                cab_booking_details[0][9],
                cab_booking_details[0][11],
                cab_booking_details[0][13],
                cab_booking_details[0][14]

            ]]


            # ==================================
            # CATEGORICAL FEATURES
            # ==================================

            x_c = ohe.transform([[

                cab_booking_details[0][0],
                cab_booking_details[0][1],
                cab_booking_details[0][2],
                cab_booking_details[0][3],
                cab_booking_details[0][4],
                cab_booking_details[0][10],
                cab_booking_details[0][12]

            ]])


            # ==================================
            # COMBINE FEATURES
            # ==================================

            if hasattr(x_c, 'toarray'):
                x_c = x_c.toarray()

            input_data = np.hstack([x_n, x_c])


            # ==================================
            # DECISION TREE REGRESSOR
            # PREDICTS TWO Y VALUES
            # ==================================

            y_reg = dtr.predict(input_data)

            y_reg = np.asarray(y_reg)

            official_fair_price = round(
                float(y_reg[0][0])
            )

            max_reasonable_seasonal_fare = round(
                float(y_reg[0][1])
            )


            # ==================================
            # DECISION TREE CLASSIFIER
            # PREDICTS ONE Y VALUE
            # ==================================

            y_clf = dtc.predict(input_data)

            quote_fairness_label = y_clf[0]


            # ==================================
            # STORE ALL THREE PREDICTIONS
            # ==================================

            prediction = {

                'official_fair_price_inr':
                    official_fair_price,

                'max_reasonable_seasonal_fare_inr':
                    max_reasonable_seasonal_fare,

                'quote_fairness_label':
                    quote_fairness_label,

                'quoted_per_person_inr':
                    quoted_per_person_inr

            }


        except Exception as e:

            error_message = str(e)


    return render_template(
        'index.html',
        prediction=prediction,
        error=error_message
    )


# ==========================================
# 3. RUN FLASK APPLICATION
# ==========================================

if __name__ == '__main__':

    app.run(debug=True)
