from flask import Flask, render_template, request
from joblib import load
import json
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle
import pandas as pd #Used to hold the data and perform different sql operations.
import numpy as np #Used to store arrays and perform operations on it.
from geopy.geocoders import Nominatim #Nominatim uses OpenStreetMap data to find locations on Earth by name and address (geocoding).
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L12-v2')
geolocator=Nominatim(user_agent="app")
locations_with_rest_coor = pd.read_csv('app/locations_with_rest_coor.csv')
with open('app/unique_cuisines_for_each_rest.p', 'rb') as fp:
    unique_cuisines_for_each_rest = pickle.load(fp)

with open('app/list_of_overall_unique_cuisines.json', 'r') as fp:
    list_of_overall_unique_cuisines = json.load(fp)

k_means_model = load('app/k_means_model.joblib')

app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def predict():
    request_type = request.method
    if request_type == 'GET':
        return render_template('index.html')
    else:

        user_cuisines = request.form['input_cuisines']
        user_cuisines = [cuisine.strip() for cuisine in user_cuisines.split(',')]
        user_location = request.form['user_location']
        user_cuisines = find_existing_cuisines_from_input(user_cuisines, model, list_of_overall_unique_cuisines['unique_cuisines'])
        if len(user_cuisines) != 0:
            df_recommends = recommend(unique_cuisines_for_each_rest, user_cuisines)
            if df_recommends.empty:
                res = "No Recommendations found!"
            else:
                input_location = user_location
                input_location_coor = calculate_user_geo_location(input_location, geolocator)
                if not pd.isna(input_location_coor):
                    res = recommend_nearest_restaurants(locations_with_rest_coor, df_recommends,
                                                                    input_location_coor, k_means_model)
                else:
                    if not df_recommends.empty:
                        res = df_recommends.iloc[0:10,:]
                        res.reset_index(inplace=True, drop=True)

                    else:
                        res = "No recommendations found!"
        else:
            res = "No restaurants found!"
        #return render_template('index.html',recommendation="The Recommendations are: {}".format(res_str.to_html()))
        if isinstance(res,pd.DataFrame):
            res = res.to_html()
        else:
            res = "<h1>" + res + "</h1>"
        # write html to file
        text_file = open("app/templates/response.html", "w",encoding="utf-8")
        text_file.write('<body>')
        text_file.write(res)
        text_file.write('</body>')
        text_file.close()
        return render_template('response.html')



def find_existing_cuisines_from_input(input_cuisines,model,list_of_overall_unique_cuisines): #find existing cuisines, new cuisines and process them accordingly
    list_find_cuisines_for = []
    cuis_to_be_added = []
    for ic in input_cuisines:
        if ic not in list_of_overall_unique_cuisines:
            list_find_cuisines_for.append(ic)
            found_cuis = get_similar_cuisine(model,ic,list_of_overall_unique_cuisines)

            if not(found_cuis is None):
                cuis_to_be_added = cuis_to_be_added + found_cuis
    input_cuisines = [ic for ic in input_cuisines if ic not in list_find_cuisines_for]
    if len(cuis_to_be_added) != 0:
        input_cuisines = input_cuisines + cuis_to_be_added
    input_cuisines = list(set(input_cuisines))
    return input_cuisines


def get_similar_cuisine(model,ic,list_of_overall_unique_cuisines):#Using Transformer to get the similar existing cuisines based on the input cuisines
    cuis_score_df=pd.DataFrame(columns=['cuisines','scores'])
    for cuis in list_of_overall_unique_cuisines:
        cuis_score_dict={}
        ic_vector = model.encode(ic)
        cuis_vector = model.encode(cuis)
        score = util.cos_sim(ic_vector,cuis_vector)
        cuis_score_dict['cuisines'] = cuis
        cuis_score_dict['scores'] = float(score)
        cuis_score_df = cuis_score_df.append(cuis_score_dict,ignore_index=True)
    cuis_score_df = cuis_score_df[cuis_score_df['scores'] > 0.45]
    if not cuis_score_df.empty:
        return list(cuis_score_df['cuisines'].values)
    else:
        return None


def recommend(unique_cuisines_for_each_rest,user_cuisines): #Recommending restaurants based on cuisines and ratings
    input_cuisines = user_cuisines
    input_cuisines = [cuisine.lower() for cuisine in input_cuisines]
    df_recommends=pd.DataFrame(columns=['restaurant','restaurant_ratings','present_cuisines_count','present_cuisines'])
    for rest,values in unique_cuisines_for_each_rest.items():
        recommend_rest_dict={}
        rest_cuisines = [cuisine.lower() for cuisine in values[1]]
        present_cuisines = [cuisine for cuisine in input_cuisines if cuisine in rest_cuisines] #Get only the cuisines from the restaurant which is in input list of cuisines
        present_cuisines_count = len(present_cuisines)
        if (present_cuisines_count > 0) and (values[0] >= 3.5): # Recommending restaurants which contains cuisines > 0 and ratings >= 3.5
            recommend_rest_dict['restaurant'] = rest
            recommend_rest_dict['present_cuisines_count'] = present_cuisines_count
            recommend_rest_dict['present_cuisines'] = present_cuisines
            recommend_rest_dict['restaurant_ratings'] = values[0]
            recommend_rest_dict['location'] = values[2]
            df_recommends = df_recommends.append(recommend_rest_dict,ignore_index=True)
    #df_recommends.drop(['present_cuisines_count'],axis=1,inplace=True)
    return df_recommends


def calculate_user_geo_location(location,geolocator):#Function to Calculate the geo-locations - Latitude and Longitude for input user location
    location = geolocator.geocode(location)
    if location is None:
        return np.nan
    else:
        geo = (location.latitude,location.longitude)
        return geo


def recommend_nearest_restaurants(locations_with_rest_coor,df_recommends,input_location_coor,k_model): #Recommend restaurants based on the distance neares to the user location
    #list_of_dist = []
    input_loc_cluster = k_model.predict([[input_location_coor[0],input_location_coor[1]]])[0]
    locations_with_rest_coor_clus = locations_with_rest_coor[locations_with_rest_coor['cluster'] == input_loc_cluster]
    df_recommends_final = df_recommends[df_recommends['location'].apply(lambda x: True if x in locations_with_rest_coor_clus['location'] else False)]
    if df_recommends_final.empty:
        df_recommends_final = df_recommends
    df_recommends_final = df_recommends_final.iloc[0:10,:]
    df_recommends_final.reset_index(inplace=True,drop=True)
    df_recommends_final['present_cuisines'] = df_recommends['present_cuisines'].apply(lambda x: ','.join(x))
    df_recommends_final.sort_values(by=['present_cuisines_count'], ascending=False, inplace=True)
    df_recommends_final.drop(['present_cuisines_count'], axis=1, inplace=True)
    return df_recommends_final
