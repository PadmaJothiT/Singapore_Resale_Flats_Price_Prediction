import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import unicodedata
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

#Loading the models
xgb_model_path = r"C:\Users\Padma Jothi\Desktop\Capstone\Singapore Resale\XGBRegressor.pkl"

with open(xgb_model_path, 'rb') as file:
    xgb_model = pickle.load(file)

#Scaling
scaling_flat = r"C:\Users\Padma Jothi\Desktop\Capstone\Singapore Resale\scaling_sing.pkl"

with open (scaling_flat,"rb") as file_1:
    scaler_flat = pickle.load(file_1)


# Streamlit App
icon = Image.open(r"C:\Users\Padma Jothi\Desktop\Capstone\Singapore Resale\singapore.jpeg")
st.set_page_config(page_title="Singapore Housing Price Predictor", layout="wide", page_icon=icon)

# Header and description
st.markdown("<h1 style='text-align: center; color: gold;'>SINGAPORE RESALE FLAT PRICE PREDICTOR</h1>", unsafe_allow_html=True)
st.write('***Welcome to the Singapore Resale Flat Price Prediction webpage. Machine Learning algorithms are used to predict the resale flat prices. The value of the price is predicted in Singapore dollars.***')

col1,col2,col3 = st.columns(3)


with col1:
    years = st.selectbox("Select the Year", [str(i) for i in range(1990, 2025)])
    
    months = st.selectbox("Select the Month", [str(i) for i in range(1, 13)])
    
    town = st.selectbox("Select the Town", ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA',
                                            'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA','LIM CHU KANG', 'MARINE PARADE', 
                                            'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG','SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
    
    flat_type = st.selectbox("Select the Flat Type", ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'])

with col2: 
    street_name = st.selectbox("Select the Street Name",[
        'ANG MO KIO AVE 1', 'ANG MO KIO AVE 3', 'ANG MO KIO AVE 4',
    'ANG MO KIO AVE 10', 'ANG MO KIO AVE 5', 'ANG MO KIO AVE 8',
    'ANG MO KIO AVE 6', 'ANG MO KIO AVE 9', 'ANG MO KIO AVE 2',
    'BEDOK RESERVOIR RD', 'BEDOK NTH ST 3', 'BEDOK STH RD',
    'NEW UPP CHANGI RD', 'BEDOK NTH RD', 'BEDOK STH AVE 1',
    'CHAI CHEE RD', 'CHAI CHEE DR', 'BEDOK NTH AVE 4',
    'BEDOK STH AVE 3', 'BEDOK STH AVE 2', 'BEDOK NTH ST 2',
    'BEDOK NTH ST 4', 'BEDOK NTH AVE 2', 'BEDOK NTH AVE 3',
    'BEDOK NTH AVE 1', 'BEDOK NTH ST 1', 'CHAI CHEE ST', 'SIN MING RD',
    'SHUNFU RD', 'BT BATOK ST 11', 'BT BATOK WEST AVE 8',
    'BT BATOK WEST AVE 6', 'BT BATOK ST 21', 'BT BATOK EAST AVE 5',
    'BT BATOK EAST AVE 4', 'HILLVIEW AVE', 'BT BATOK CTRL',
    'BT BATOK ST 31', 'BT BATOK EAST AVE 3', 'TAMAN HO SWEE',
    'TELOK BLANGAH CRES', 'BEO CRES', 'TELOK BLANGAH DR', 'DEPOT RD',
    'TELOK BLANGAH RISE', 'JLN BT MERAH', 'HENDERSON RD', 'INDUS RD',
    'BT MERAH VIEW', 'HENDERSON CRES', 'BT PURMEI RD',
    'TELOK BLANGAH HTS', 'EVERTON PK', 'KG BAHRU HILL', 'REDHILL CL',
    'HOY FATT RD', 'HAVELOCK RD', 'JLN KLINIK', 'JLN RUMAH TINGGI',
    'JLN BT HO SWEE', 'KIM CHENG ST', 'MOH GUAN TER',
    'TELOK BLANGAH WAY', 'KIM TIAN RD', 'KIM TIAN PL', 'EMPRESS RD',
    "QUEEN S RD", 'FARRER RD', 'JLN KUKOH', 'OUTRAM PK', 'SHORT ST',
    'SELEGIE RD', 'UPP CROSS ST', 'WATERLOO ST', 'QUEEN ST',
    'BUFFALO RD', 'ROWELL RD', 'ROCHOR RD', 'BAIN ST', 'SMITH ST',
    'VEERASAMY RD', 'TECK WHYE AVE', 'TECK WHYE LANE',
    'CLEMENTI AVE 3', 'WEST COAST DR', 'CLEMENTI AVE 2',
    'CLEMENTI AVE 5', 'CLEMENTI AVE 4', 'CLEMENTI AVE 1',
    'WEST COAST RD', 'CLEMENTI WEST ST 1', 'CLEMENTI WEST ST 2',
    'CLEMENTI ST 13', "C WEALTH AVE WEST", 'CLEMENTI AVE 6',
    'CLEMENTI ST 14', 'CIRCUIT RD', 'MACPHERSON LANE',
    'JLN PASAR BARU', 'GEYLANG SERAI', 'EUNOS CRES', 'SIMS DR',
    'ALJUNIED CRES', 'GEYLANG EAST AVE 1', 'DAKOTA CRES', 'PINE CL',
    'HAIG RD', 'BALAM RD', 'JLN DUA', 'GEYLANG EAST CTRL',
    'EUNOS RD 5', 'HOUGANG AVE 3', 'HOUGANG AVE 5', 'HOUGANG AVE 1',
    'HOUGANG ST 22', 'HOUGANG AVE 10', 'LOR AH SOO', 'HOUGANG ST 11',
    'HOUGANG AVE 7', 'HOUGANG ST 21', 'TEBAN GDNS RD',
    'JURONG EAST AVE 1', 'JURONG EAST ST 32', 'JURONG EAST ST 13',
    'JURONG EAST ST 21', 'JURONG EAST ST 24', 'JURONG EAST ST 31',
    'PANDAN GDNS', 'YUNG KUANG RD', 'HO CHING RD', 'HU CHING RD',
    'BOON LAY DR', 'BOON LAY AVE', 'BOON LAY PL', 'JURONG WEST ST 52',
    'JURONG WEST ST 41', 'JURONG WEST AVE 1', 'JURONG WEST ST 42',
    'JLN BATU', "ST. GEORGE S RD", 'NTH BRIDGE RD', 'FRENCH RD',
    'BEACH RD', 'WHAMPOA DR', 'UPP BOON KENG RD', 'BENDEMEER RD',
    'WHAMPOA WEST', 'LOR LIMAU', 'KALLANG BAHRU', 'GEYLANG BAHRU',
    'DORSET RD', 'OWEN RD', 'KG ARANG RD', 'JLN BAHAGIA',
    'MOULMEIN RD', 'TOWNER RD', 'JLN RAJAH', 'KENT RD', 'AH HOOD RD',
    "KING GEORGE S AVE", 'CRAWFORD LANE', 'MARINE CRES', 'MARINE DR',
    'MARINE TER', "C WEALTH CL", "C WEALTH DR", 'TANGLIN HALT RD',
    "C WEALTH CRES", 'DOVER RD', 'MARGARET DR', 'GHIM MOH RD',
    'DOVER CRES', 'STIRLING RD', 'MEI LING ST', 'HOLLAND CL',
    'HOLLAND AVE', 'HOLLAND DR', 'DOVER CL EAST',
    'SELETAR WEST FARMWAY 6', 'LOR LEW LIAN', 'SERANGOON NTH AVE 1',
    'SERANGOON AVE 2', 'SERANGOON AVE 4', 'SERANGOON CTRL',
    'TAMPINES ST 11', 'TAMPINES ST 21', 'TAMPINES ST 91',
    'TAMPINES ST 81', 'TAMPINES AVE 4', 'TAMPINES ST 22',
    'TAMPINES ST 12', 'TAMPINES ST 23', 'TAMPINES ST 24',
    'TAMPINES ST 41', 'TAMPINES ST 82', 'TAMPINES ST 83',
    'TAMPINES AVE 5', 'LOR 2 TOA PAYOH', 'LOR 8 TOA PAYOH',
    'LOR 1 TOA PAYOH', 'LOR 5 TOA PAYOH', 'LOR 3 TOA PAYOH',
    'LOR 7 TOA PAYOH', 'TOA PAYOH EAST', 'LOR 4 TOA PAYOH',
    'TOA PAYOH CTRL', 'TOA PAYOH NTH', 'POTONG PASIR AVE 3',
    'POTONG PASIR AVE 1', 'UPP ALJUNIED LANE', 'JOO SENG RD',
    'MARSILING LANE', 'MARSILING DR', 'MARSILING RISE',
    'MARSILING CRES', 'WOODLANDS CTR RD', 'WOODLANDS ST 13',
    'WOODLANDS ST 11', 'YISHUN RING RD', 'YISHUN AVE 5',
    'YISHUN ST 72', 'YISHUN ST 11', 'YISHUN ST 21', 'YISHUN ST 22',
    'YISHUN AVE 3', 'CHAI CHEE AVE', 'ZION RD', 'LENGKOK BAHRU',
    'SPOTTISWOODE PK RD', 'NEW MKT RD', 'TG PAGAR PLAZA',
    'KELANTAN RD', 'PAYA LEBAR WAY', 'UBI AVE 1', 'SIMS AVE',
    'YUNG PING RD', 'TAO CHING RD', 'GLOUCESTER RD', 'BOON KENG RD',
    'WHAMPOA STH', 'CAMBRIDGE RD', 'TAMPINES ST 42', 'LOR 6 TOA PAYOH',
    'KIM KEAT AVE', 'YISHUN AVE 6', 'YISHUN AVE 9', 'YISHUN ST 71',
    'BT BATOK ST 32', 'SILAT AVE', 'TIONG BAHRU RD', 'SAGO LANE',
    "ST. GEORGE S LANE", 'LIM CHU KANG RD', "C WEALTH AVE",
    "QUEEN S CL", 'SERANGOON AVE 3', 'POTONG PASIR AVE 2',
    'WOODLANDS AVE 1', 'YISHUN AVE 4', 'LOWER DELTA RD', 'NILE RD',
    'JLN MEMBINA BARAT', 'JLN BERSEH', 'CHANDER RD', 'CASSIA CRES',
    'OLD AIRPORT RD', 'ALJUNIED RD', 'BUANGKOK STH FARMWAY 1',
    'BT BATOK ST 33', 'ALEXANDRA RD', 'CHIN SWEE RD', 'SIMS PL',
    'HOUGANG AVE 2', 'HOUGANG AVE 8', 'SEMBAWANG RD', 'SIMEI ST 1',
    'BT BATOK ST 34', 'BT MERAH CTRL', 'LIM LIAK ST', 'JLN TENTERAM',
    'WOODLANDS ST 32', 'SIN MING AVE', 'BT BATOK ST 52', 'DELTA AVE',
    'PIPIT RD', 'HOUGANG AVE 4', 'QUEENSWAY', 'YISHUN ST 61',
    'BISHAN ST 12', "JLN MA MOR", 'TAMPINES ST 44', 'TAMPINES ST 43',
    'BISHAN ST 13', 'JLN DUSUN', 'YISHUN AVE 2', 'JOO CHIAT RD',
    'EAST COAST RD', 'REDHILL RD', 'KIM PONG RD', 'RACE COURSE RD',
    'KRETA AYER RD', 'HOUGANG ST 61', 'TESSENSOHN RD', 'MARSILING RD',
    'YISHUN ST 81', 'BT BATOK ST 51', 'BT BATOK WEST AVE 4',
    'BT BATOK WEST AVE 2', 'JURONG WEST ST 91', 'JURONG WEST ST 81',
    'GANGSA RD', 'MCNAIR RD', 'SIMEI ST 4', 'YISHUN AVE 7',
    'SERANGOON NTH AVE 2', 'YISHUN AVE 11', 'BANGKIT RD',
    'JURONG WEST ST 73', 'OUTRAM HILL', 'HOUGANG AVE 6',
    'PASIR RIS ST 12', 'PENDING RD', 'PETIR RD', 'LOR 3 GEYLANG',
    'BISHAN ST 11', 'PASIR RIS DR 6', 'BISHAN ST 23',
    'JURONG WEST ST 92', 'PASIR RIS ST 11', 'YISHUN CTRL',
    'BISHAN ST 22', 'SIMEI RD', 'TAMPINES ST 84', 'BT PANJANG RING RD',
    'JURONG WEST ST 93', 'FAJAR RD', 'WOODLANDS ST 81',
    'CHOA CHU KANG CTRL', 'PASIR RIS ST 51', 'HOUGANG ST 52',
    'CASHEW RD', 'TOH YI DR', 'HOUGANG CTRL', 'KG KAYU RD',
    'TAMPINES AVE 8', 'TAMPINES ST 45', 'SIMEI ST 2',
    'WOODLANDS AVE 3', 'LENGKONG TIGA', 'WOODLANDS ST 82',
    'SERANGOON NTH AVE 4', 'SERANGOON CTRL DR', 'BRIGHT HILL DR',
    'SAUJANA RD', 'CHOA CHU KANG AVE 3', 'TAMPINES AVE 9',
    'JURONG WEST ST 51', 'YUNG HO RD', 'SERANGOON AVE 1',
    'PASIR RIS ST 41', 'GEYLANG EAST AVE 2', 'CHOA CHU KANG AVE 2',
    'KIM KEAT LINK', 'PASIR RIS DR 4', 'PASIR RIS ST 21',
    'SENG POH RD', 'HOUGANG ST 51', 'JURONG WEST ST 72',
    'JURONG WEST ST 71', 'PASIR RIS ST 52', 'TAMPINES ST 32',
    'CHOA CHU KANG AVE 4', 'CHOA CHU KANG LOOP', 'JLN TENAGA',
    'TAMPINES CTRL 1', 'TAMPINES ST 33', 'BT BATOK WEST AVE 7',
    'JURONG WEST AVE 5', 'TAMPINES AVE 7', 'WOODLANDS ST 83',
    'CHOA CHU KANG ST 51', 'PASIR RIS DR 3', 'YISHUN CTRL 1',
    'CHOA CHU KANG AVE 1', 'WOODLANDS ST 31', 'BT MERAH LANE 1',
    'PASIR RIS ST 13', 'ELIAS RD', 'BISHAN ST 24', 'WHAMPOA RD',
    'WOODLANDS ST 41', 'PASIR RIS ST 71', 'JURONG WEST ST 74',
    'PASIR RIS DR 1', 'PASIR RIS ST 72', 'PASIR RIS DR 10',
    'CHOA CHU KANG ST 52', 'CLARENCE LANE', 'CHOA CHU KANG NTH 6',
    'PASIR RIS ST 53', 'CHOA CHU KANG NTH 5', 'ANG MO KIO ST 21',
    'JLN DAMAI', 'CHOA CHU KANG ST 62', 'WOODLANDS AVE 5',
    'WOODLANDS DR 50', 'CHOA CHU KANG ST 53', 'TAMPINES ST 72',
    'UPP SERANGOON RD', 'JURONG WEST ST 75', 'STRATHMORE AVE',
    'ANG MO KIO ST 31', 'TAMPINES ST 34', 'YUNG AN RD',
    'WOODLANDS AVE 4', 'CHOA CHU KANG NTH 7', 'ANG MO KIO ST 11',
    'WOODLANDS AVE 9', 'YUNG LOH RD', 'CHOA CHU KANG DR',
    'CHOA CHU KANG ST 54', 'REDHILL LANE', 'KANG CHING RD',
    'TAH CHING RD', 'SIMEI ST 5', 'WOODLANDS DR 40', 'WOODLANDS DR 70',
    'TAMPINES ST 71', 'WOODLANDS DR 42', 'SERANGOON NTH AVE 3',
    'JELAPANG RD', 'BT BATOK ST 22', 'HOUGANG ST 91',
    'WOODLANDS AVE 6', 'WOODLANDS CIRCLE', 'CORPORATION DR',
    'LOMPANG RD', 'WOODLANDS DR 72', 'CHOA CHU KANG ST 64',
    'BT BATOK ST 24', 'JLN TECK WHYE', 'WOODLANDS CRES',
    'WOODLANDS DR 60', 'CHANGI VILLAGE RD', 'BT BATOK ST 25',
    'HOUGANG AVE 9', 'JURONG WEST CTRL 1', 'WOODLANDS RING RD',
    'CHOA CHU KANG AVE 5', 'TOH GUAN RD', 'JURONG WEST ST 61',
    'WOODLANDS DR 14', 'HOUGANG ST 92', 'CHOA CHU KANG CRES',
    'SEMBAWANG CL', 'CANBERRA RD', 'SEMBAWANG CRES', 'SEMBAWANG VISTA',
    'COMPASSVALE WALK', 'RIVERVALE ST', 'WOODLANDS DR 62',
    'SEMBAWANG DR', 'WOODLANDS DR 53', 'WOODLANDS DR 52',
    'RIVERVALE WALK', 'COMPASSVALE LANE', 'RIVERVALE DR', 'SENJA RD',
    'JURONG WEST ST 65', 'RIVERVALE CRES', 'WOODLANDS DR 44',
    'COMPASSVALE DR', 'WOODLANDS DR 16', 'COMPASSVALE RD',
    'WOODLANDS DR 73', 'HOUGANG ST 31', 'JURONG WEST ST 64',
    'WOODLANDS DR 71', 'YISHUN ST 20', 'ADMIRALTY DR',
    'COMPASSVALE ST', 'BEDOK RESERVOIR VIEW', 'YUNG SHENG RD',
    'ADMIRALTY LINK', 'SENGKANG EAST WAY', 'ANG MO KIO ST 32',
    'ANG MO KIO ST 52', 'BOON TIONG RD', 'JURONG WEST ST 62',
    'ANCHORVALE LINK', 'CANBERRA LINK', 'COMPASSVALE CRES',
    'CLEMENTI ST 12', 'MONTREAL DR', 'WELLINGTON CIRCLE',
    'SENGKANG EAST RD', 'JURONG WEST AVE 3', 'ANCHORVALE LANE',
    'SENJA LINK', 'EDGEFIELD PLAINS', 'ANCHORVALE DR', 'SEGAR RD',
    'FARRER PK RD', 'PUNGGOL FIELD', 'EDGEDALE PLAINS',
    'ANCHORVALE RD', 'CANTONMENT CL', 'JLN MEMBINA', 'FERNVALE LANE',
    'JURONG WEST ST 25', 'CLEMENTI ST 11', 'PUNGGOL FIELD WALK',
    'KLANG LANE', 'PUNGGOL CTRL', 'JELEBU RD', 'BUANGKOK CRES',
    'WOODLANDS DR 75', 'BT BATOK WEST AVE 5', 'JELLICOE RD',
    'PUNGGOL DR', 'JURONG WEST ST 24', 'SEMBAWANG WAY', 'FERNVALE RD',
    'BUANGKOK LINK', 'FERNVALE LINK', 'JLN TIGA', 'YUAN CHING RD',
    'COMPASSVALE LINK', 'MARINE PARADE CTRL', 'COMPASSVALE BOW',
    'PUNGGOL RD', 'BEDOK CTRL', 'PUNGGOL EAST', 'SENGKANG CTRL',
    'TAMPINES CTRL 7', 'SENGKANG WEST AVE', 'PUNGGOL PL',
    'CANTONMENT RD', 'GHIM MOH LINK', 'SIMEI LANE', 'YISHUN ST 41',
    'TELOK BLANGAH ST 31', 'JLN KAYU', 'LOR 1A TOA PAYOH',
    'PUNGGOL WALK', 'SENGKANG WEST WAY', 'BUANGKOK GREEN',
    'PUNGGOL WAY', 'YISHUN ST 31', 'TECK WHYE CRES', 'MONTREAL LINK',
    'UPP SERANGOON CRES', 'SUMANG LINK', 'SENGKANG EAST AVE',
    'YISHUN AVE 1', 'ANCHORVALE CRES', 'ANCHORVALE ST',
    'TAMPINES CTRL 8', 'YISHUN ST 51', 'UPP SERANGOON VIEW',
    'TAMPINES AVE 1', 'BEDOK RESERVOIR CRES', 'ANG MO KIO ST 61',
    'DAWSON RD', 'FERNVALE ST', 'HOUGANG ST 32', 'TAMPINES ST 86',
    'SUMANG WALK', 'CHOA CHU KANG AVE 7', 'KEAT HONG CL',
    'JURONG WEST CTRL 3', 'KEAT HONG LINK', 'ALJUNIED AVE 2',
    'CANBERRA CRES', 'SUMANG LANE', 'CANBERRA ST', 'ANG MO KIO ST 44',
    'ANG MO KIO ST 51', 'BT BATOK EAST AVE 6', 'BT BATOK WEST AVE 9',
    'CANBERRA WALK', 'WOODLANDS RISE', 'TAMPINES ST 61'
    ])
    
    block = st.text_input('**Enter the block number (eg.201A)**', value=254)
    # Define a mapping for letters to decimal values
    letter_mapping = {chr(ord('A') + i): f'.{i + 1}' for i in range(26)}
    block_decimal = float(''.join(letter_mapping.get(c, c) for c in block))
    
    flr_area_sqm = st.number_input("Enter the **:red[Floor Area]** (sqm): min_value=31, max_value=280, step=1")
    
    flat_model = st.selectbox("Select the Flat Model", [
        'IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED', 'PREMIUM APARTMENT', 'MAISONETTE',
        'APARTMENT', 'MODEL A2', 'TYPE S1', 'TYPE S2', 'ADJOINED FLAT', 'TERRACE', 'DBSS', 'MODEL-A-MAISONETTE',
        'Premium Maisonette', 'MULTI GENERATION', 'Premium Apartment Loft', 'IMPROVED-MAISONETTE', '2-ROOM', '3Gen'
    ])
    
with col3:
    stry_start= st.number_input("Enter the **:red[Storey Start]**: min_value=1, max_value=50")

    stry_end= st.number_input("Enter the **:red[Storey End]**: min_value=1, max_value=50")
    
    re_les_year = st.number_input("Enter the **:red[Remaining Lease]** (years):, min_value=41, max_value=97, step=1")
    
    re_les_month = st.selectbox("Select the Remaining Lease in Months",[
        '7',  '4',  '5',  '6',  '8', '10',  '3',  '9', '11',  '2',  '0'])
    
    les_coms_dt = st.selectbox("Select the Lease Commencement Date", [str(i) for i in range(1966, 2023)])



#Mapping encoded values to original values
def town_mapping(town):
    town_dict ={ 'ANG MO KIO' : 0,'BEDOK' : 1,'BISHAN' : 2,'BUKIT BATOK' : 3,
        'BUKIT MERAH' : 4,'BUKIT TIMAH' : 6,'CENTRAL AREA' : 7,'CHOA CHU KANG' : 8,
        'CLEMENTI' : 9,'GEYLANG' : 10,'HOUGANG' : 11,'JURONG EAST' : 12,'JURONG WEST' : 13,
        'KALLANG/WHAMPOA' : 14,'MARINE PARADE' : 16,'QUEENSTOWN' : 19,'SENGKANG' : 21,
        'SERANGOON' : 22,'TAMPINES' : 23,'TOA PAYOH' : 24,'WOODLANDS' : 25,'YISHUN' : 26,
        'LIM CHU KANG' : 15,'SEMBAWANG' : 20,'BUKIT PANJANG' : 5,'PASIR RIS' : 17,'PUNGGOL' : 18
        }
    return town_dict[town]


def flat_type_mapping(flat_type):
    flat_type_dict ={'1 ROOM' : 0,'3 ROOM' : 2,'4 ROOM' : 3,'5 ROOM' : 4,
                    '2 ROOM' : 1,'EXECUTIVE' : 5,'MULTI GENERATION' : 6
                    }
    return flat_type_dict[flat_type]

def flat_model_mapping(flat_model):
    flat_model_dict = {'IMPROVED' : 5,'NEW GENERATION' : 12,'MODEL A' : 8,'STANDARD' : 17,'SIMPLIFIED' : 16,
                    'MODEL A-MAISONETTE' : 9,'APARTMENT' : 2,'MAISONETTE' : 7,'TERRACE' : 18,'2-ROOM' : 0,
                    'IMPROVED-MAISONETTE' : 6,'MULTI GENERATION' : 10,'PREMIUM APARTMENT' : 13,'Adjoined flat' : 3,
                    'Premium Maisonette' : 15,'Model A2' : 11,'DBSS' : 4,'Type S1' : 19,'Type S2' : 20,
                    'Premium Apartment Loft' : 14,'3Gen' : 1
                    }
    return flat_model_dict[flat_model]

def street_name_mapping(street_name):
    street_name_dict = {'ANG MO KIO AVE 1' : 13,'ANG MO KIO AVE 3' : 16,'ANG MO KIO AVE 4' : 17,'ANG MO KIO AVE 10' : 14,
                        'ANG MO KIO AVE 5' : 18,'ANG MO KIO AVE 8' : 20,'ANG MO KIO AVE 6' : 19,'ANG MO KIO AVE 9' : 21,
                        'ANG MO KIO AVE 2' : 15,'BEDOK RESERVOIR RD' : 45,'BEDOK NTH ST 3' : 42,'BEDOK STH RD' : 50,
                        'NEW UPP CHANGI RD' : 340,'BEDOK NTH RD' : 39,'BEDOK STH AVE 1' : 47,'CHAI CHEE RD' : 115,
                        'CHAI CHEE DR' : 114,'BEDOK NTH AVE 4' : 38,'BEDOK STH AVE 3' : 49,'BEDOK STH AVE 2' : 48,
                        'BEDOK NTH ST 2' : 41,'BEDOK NTH ST 4' : 43,'BEDOK NTH AVE 2' : 36,'BEDOK NTH AVE 3' : 37,
                        'BEDOK NTH AVE 1' : 35,'BEDOK NTH ST 1' : 40,'CHAI CHEE ST' : 116,'SIN MING RD' : 437,
                        'SHUNFU RD' : 425,'BT BATOK ST 11' : 70,'BT BATOK WEST AVE 8' : 86,'BT BATOK WEST AVE 6' : 84,
                        'BT BATOK ST 21' : 71,'BT BATOK EAST AVE 5' : 68,'BT BATOK EAST AVE 4' : 67,'HILLVIEW AVE' : 200,
                        'BT BATOK CTRL' : 65,'BT BATOK ST 31' : 75,'BT BATOK EAST AVE 3' : 66,'TAMAN HO SWEE' : 448,
                        'TELOK BLANGAH CRES' : 487,'BEO CRES' : 52,'TELOK BLANGAH DR' : 488,'DEPOT RD' : 166,
                        'TELOK BLANGAH RISE' : 490,'JLN BT MERAH' : 236,'HENDERSON RD' : 199,'INDUS RD' : 228,
                        'BT MERAH VIEW' : 90,'HENDERSON CRES' : 198,'BT PURMEI RD' : 92,'TELOK BLANGAH HTS' : 489,
                        'EVERTON PK' : 178,'KG BAHRU HILL' : 292,'REDHILL CL' : 385,'HOY FATT RD' : 226,'HAVELOCK RD' : 197,
                        'JLN KLINIK' : 241,'JLN RUMAH TINGGI' : 248,'JLN BT HO SWEE' : 235,'KIM CHENG ST' : 294,
                        'MOH GUAN TER' : 335,'TELOK BLANGAH WAY' : 492,'KIM TIAN RD' : 299,'KIM TIAN PL' : 298,
                        'EMPRESS RD' : 175,'QUEEN S RD' : 382,'FARRER RD' : 181,'JLN KUKOH' : 242,'OUTRAM PK' : 345,
                        'SHORT ST' : 424,'SELEGIE RD' : 397,'UPP CROSS ST' : 505,'WATERLOO ST' : 510,'QUEEN ST' : 380,
                        'BUFFALO RD' : 97,'ROWELL RD' : 393,'ROCHOR RD' : 392,'BAIN ST' : 30,'SMITH ST' : 438,
                        'VEERASAMY RD' : 509,'TECK WHYE AVE' : 484,'TECK WHYE LANE' : 486,'CLEMENTI AVE 3' : 143,
                        'WEST COAST DR' : 512,'CLEMENTI AVE 2' : 142,'CLEMENTI AVE 5' : 145,'CLEMENTI AVE 4' : 144,
                        'CLEMENTI AVE 1' : 141,'WEST COAST RD' : 513,'CLEMENTI WEST ST 1' : 151,'CLEMENTI WEST ST 2' : 152,
                        'CLEMENTI ST 13' : 149,'C WEALTH AVE WEST' : 99,'CLEMENTI AVE 6' : 146,'CLEMENTI ST 14' : 150,
                        'CIRCUIT RD' : 139,'MACPHERSON LANE' : 322,'JLN PASAR BARU' : 246,'GEYLANG SERAI' : 192,
                        'EUNOS CRES' : 176,'SIMS DR' : 434,'ALJUNIED CRES' : 5,'GEYLANG EAST AVE 1' : 189,'DAKOTA CRES' : 163,
                        'PINE CL' : 366,'HAIG RD' : 196,'BALAM RD' : 31,'JLN DUA' : 238,'GEYLANG EAST CTRL' : 191,'EUNOS RD 5' : 177,
                        'HOUGANG AVE 3' : 208,'HOUGANG AVE 5' : 210,'HOUGANG AVE 1' : 205,'HOUGANG ST 22' : 218,'HOUGANG AVE 10' : 206,
                        'LOR AH SOO' : 318,'HOUGANG ST 11' : 216,'HOUGANG AVE 7' : 212,'HOUGANG ST 21' : 217,'TEBAN GDNS RD' : 483,
                        'JURONG EAST AVE 1' : 255,'JURONG EAST ST 32' : 260,'JURONG EAST ST 13' : 256,'JURONG EAST ST 21' : 257,
                        'JURONG EAST ST 24' : 258,  'JURONG EAST ST 31' : 259,'PANDAN GDNS' : 347,'YUNG KUANG RD' : 579,
                        'HO CHING RD' : 201,'HU CHING RD' : 227,'BOON LAY DR' : 61,'BOON LAY AVE' : 60,'BOON LAY PL' : 62,
                        'JURONG WEST ST 52' : 271,'JURONG WEST ST 41' : 268,'JURONG WEST AVE 1' : 261,'JURONG WEST ST 42' : 269,
                        'JLN BATU' : 233,'ST. GEORGE S RD' : 441,'NTH BRIDGE RD' : 342,'FRENCH RD' : 186,'BEACH RD' : 33,
                        'WHAMPOA DR' : 514,'UPP BOON KENG RD' : 504,'BENDEMEER RD' : 51,'WHAMPOA WEST' : 517,'LOR LIMAU' : 320,
                        'KALLANG BAHRU' : 285,'GEYLANG BAHRU' : 188,'DORSET RD' : 167,'OWEN RD' : 346,'KG ARANG RD' : 291,
                        'JLN BAHAGIA' : 232,'MOULMEIN RD' : 338,'TOWNER RD' : 501,'JLN RAJAH' : 247,'KENT RD' : 290,
                        'AH HOOD RD' : 2,'KING GEORGE S AVE' : 300,'CRAWFORD LANE' : 162,'MARINE CRES' : 324,'MARINE DR' : 325,
                        'MARINE TER' : 327,'C WEALTH CL' : 100,'C WEALTH DR' : 102,'TANGLIN HALT RD' : 481,'C WEALTH CRES' : 101,
                        'DOVER RD' : 170,'MARGARET DR' : 323,'GHIM MOH RD' : 194,'DOVER CRES' : 169,'STIRLING RD' : 442,
                        'MEI LING ST' : 334,'HOLLAND CL' : 203,'HOLLAND AVE' : 202,'HOLLAND DR' : 204,'DOVER CL EAST' : 168,
                        'SELETAR WEST FARMWAY 6' : 398,'LOR LEW LIAN' : 319,'SERANGOON NTH AVE 1' : 420,'SERANGOON AVE 2' : 415,
                        'SERANGOON AVE 4' : 417,'SERANGOON CTRL' : 418,'TAMPINES ST 11' : 458,'TAMPINES ST 21' : 460,
                        'TAMPINES ST 91' : 480,'TAMPINES ST 81' : 475,'TAMPINES AVE 4' : 450,'TAMPINES ST 22' : 461,
                        'TAMPINES ST 12' : 459,'TAMPINES ST 23' : 462,'TAMPINES ST 24' : 463,'TAMPINES ST 41' : 467,
                        'TAMPINES ST 82' : 476,'TAMPINES ST 83' : 477,'TAMPINES AVE 5' : 451,'LOR 2 TOA PAYOH' : 310,
                        'LOR 8 TOA PAYOH' : 317,'LOR 1 TOA PAYOH' : 308,'LOR 5 TOA PAYOH' : 314,'LOR 3 TOA PAYOH' : 312,
                        'LOR 7 TOA PAYOH' : 316,'TOA PAYOH EAST' : 497,'LOR 4 TOA PAYOH' : 313,'TOA PAYOH CTRL' : 496,
                        'TOA PAYOH NTH' : 498,'POTONG PASIR AVE 3' : 370,'POTONG PASIR AVE 1' : 368,'UPP ALJUNIED LANE' : 503,
                        'JOO SENG RD' : 254,'MARSILING LANE' : 330,'MARSILING DR' : 329,'MARSILING RISE' : 332,
                        'MARSILING CRES' : 328,'WOODLANDS CTR RD' : 526,'WOODLANDS ST 13' : 545,'WOODLANDS ST 11' : 544,
                        'YISHUN RING RD' : 563,'YISHUN AVE 5' : 557,'YISHUN ST 72' : 574,'YISHUN ST 11' : 564,
                        'YISHUN ST 21' : 566,'YISHUN ST 22' : 567,'YISHUN AVE 3' : 555,'CHAI CHEE AVE' : 113,
                        'ZION RD' : 583,'LENGKOK BAHRU' : 303,'SPOTTISWOODE PK RD' : 439,'NEW MKT RD' : 339,
                        'TG PAGAR PLAZA' : 494,'KELANTAN RD' : 289,'PAYA LEBAR WAY' : 363,'UBI AVE 1' : 502,
                        'SIMS AVE' : 433,'YUNG PING RD' : 581,'TAO CHING RD' : 482,'GLOUCESTER RD' : 195,'BOON KENG RD' : 59,
                        'WHAMPOA STH' : 516,'CAMBRIDGE RD' : 103,'TAMPINES ST 42' : 468,'LOR 6 TOA PAYOH' : 315,
                        'KIM KEAT AVE' : 295,'YISHUN AVE 6' : 558,'YISHUN AVE 9' : 560,'YISHUN ST 71' : 573,'BT BATOK ST 32' : 76,
                        'SILAT AVE' : 426,'TIONG BAHRU RD' : 495,'SAGO LANE' : 394,'ST. GEORGE S LANE' : 440,'LIM CHU KANG RD' : 305,
                        'C WEALTH AVE' : 98,'QUEEN S CL' : 381,'SERANGOON AVE 3' : 416,'POTONG PASIR AVE 2' : 369,
                        'WOODLANDS AVE 1' : 518,'YISHUN AVE 4' : 556,'LOWER DELTA RD' : 321,'NILE RD' : 341,'JLN MEMBINA BARAT' : 245,
                        'JLN BERSEH' : 234,'CHANDER RD' : 117,'CASSIA CRES' : 112,'OLD AIRPORT RD' : 343,'ALJUNIED RD' : 6,
                        'BUANGKOK STH FARMWAY 1' : 96,'BT BATOK ST 33' : 77,'ALEXANDRA RD' : 3,'CHIN SWEE RD' : 119,
                        'SIMS PL' : 435,'HOUGANG AVE 2' : 207,'HOUGANG AVE 8' : 213,'SEMBAWANG RD' : 402,'SIMEI ST 1' : 429,
                        'BT BATOK ST 34' : 78,'BT MERAH CTRL' : 88,'LIM LIAK ST' : 306,'JLN TENTERAM' : 251,'WOODLANDS ST 32' : 547,
                        'SIN MING AVE' : 436,'BT BATOK ST 52' : 80,'DELTA AVE' : 165,'PIPIT RD' : 367,'HOUGANG AVE 4' : 209,'QUEENSWAY' : 383,
                        'YISHUN ST 61' : 572,'BISHAN ST 12' : 54,'JLN MA MOR' : 243,'TAMPINES ST 44' : 470,'TAMPINES ST 43' : 469,
                        'BISHAN ST 13' : 55,'JLN DUSUN' : 239,'YISHUN AVE 2' : 554,'JOO CHIAT RD' : 253,'EAST COAST RD' : 171,
                        'REDHILL RD' : 387,'KIM PONG RD' : 297,'RACE COURSE RD' : 384,'KRETA AYER RD' : 302,'HOUGANG ST 61' : 223,
                        'TESSENSOHN RD' : 493,'MARSILING RD' : 331,'YISHUN ST 81' : 575,'BT BATOK ST 51' : 79,'BT BATOK WEST AVE 4' : 82,
                        'BT BATOK WEST AVE 2' : 81,'JURONG WEST ST 91' : 282,'JURONG WEST ST 81' : 281,'GANGSA RD' : 187,'MCNAIR RD' : 333,
                        'SIMEI ST 4' : 431,'YISHUN AVE 7' : 559,'SERANGOON NTH AVE 2' : 421,'YISHUN AVE 11' : 553,'BANGKIT RD' : 32,
                        'JURONG WEST ST 73' : 278,'OUTRAM HILL' : 344,'HOUGANG AVE 6' : 211,'PASIR RIS ST 12' : 354,'PENDING RD' : 364,
                        'PETIR RD' : 365,'LOR 3 GEYLANG' : 311,'BISHAN ST 11' : 53,'PASIR RIS DR 6' : 352,'BISHAN ST 23' : 57,
                        'JURONG WEST ST 92' : 283,'PASIR RIS ST 11' : 353,'YISHUN CTRL' : 561,'BISHAN ST 22' : 56,'SIMEI RD' : 428,
                        'TAMPINES ST 84' : 478,'BT PANJANG RING RD' : 91,'JURONG WEST ST 93' : 284,'FAJAR RD' : 179,'WOODLANDS ST 81' : 549,
                        'CHOA CHU KANG CTRL' : 127,'PASIR RIS ST 51' : 358,'HOUGANG ST 52' : 222,'CASHEW RD' : 111,'TOH YI DR' : 500,
                        'HOUGANG CTRL' : 215,'KG KAYU RD' : 293,'TAMPINES AVE 8' : 453,'TAMPINES ST 45' : 471,'SIMEI ST 2' : 430,
                        'WOODLANDS AVE 3' : 519,'LENGKONG TIGA' : 304,'WOODLANDS ST 82' : 550,'SERANGOON NTH AVE 4' : 423,'SERANGOON CTRL DR' : 419,
                        'BRIGHT HILL DR' : 64,'SAUJANA RD' : 395,'CHOA CHU KANG AVE 3' : 122,'TAMPINES AVE 9' : 454,'JURONG WEST ST 51' : 270,
                        'YUNG HO RD' : 578,'SERANGOON AVE 1' : 414,'PASIR RIS ST 41' : 357,'GEYLANG EAST AVE 2' : 190,'CHOA CHU KANG AVE 2' : 121,
                        'KIM KEAT LINK' : 296,'PASIR RIS DR 4' : 351,'PASIR RIS ST 21' : 356,'SENG POH RD' : 405,'HOUGANG ST 51' : 221,
                        'JURONG WEST ST 72' : 277,'JURONG WEST ST 71' : 276,'PASIR RIS ST 52' : 359,'TAMPINES ST 32' : 464,'CHOA CHU KANG AVE 4' : 123,
                        'CHOA CHU KANG LOOP' : 129,'JLN TENAGA' : 250,'TAMPINES CTRL 1' : 455,'TAMPINES ST 33' : 465,'BT BATOK WEST AVE 7' : 85,
                        'JURONG WEST AVE 5' : 263,'TAMPINES AVE 7' : 452,'WOODLANDS ST 83' : 551,'CHOA CHU KANG ST 51' : 133,'PASIR RIS DR 3' : 350,
                        'YISHUN CTRL 1' : 562,'CHOA CHU KANG AVE 1' : 120,'WOODLANDS ST 31' : 546,'BT MERAH LANE 1' : 89,'PASIR RIS ST 13' : 355,
                        'ELIAS RD' : 174,'BISHAN ST 24' : 58,'WHAMPOA RD' : 515,'WOODLANDS ST 41' : 548,'PASIR RIS ST 71' : 361,'JURONG WEST ST 74' : 279,
                        'PASIR RIS DR 1' : 348,'PASIR RIS ST 72' : 362,'PASIR RIS DR 10' : 349,'CHOA CHU KANG ST 52' : 134,'CLARENCE LANE' : 140,
                        'CHOA CHU KANG NTH 6' : 131,'PASIR RIS ST 53' : 360,'CHOA CHU KANG NTH 5' : 130,'ANG MO KIO ST 21' : 23,'JLN DAMAI' : 237,
                        'CHOA CHU KANG ST 62' : 137,'WOODLANDS AVE 5' : 521,'WOODLANDS DR 50' : 532,'CHOA CHU KANG ST 53' : 135,'TAMPINES ST 72' : 474,
                        'UPP SERANGOON RD' : 507,'JURONG WEST ST 75' : 280,'STRATHMORE AVE' : 443,'ANG MO KIO ST 31' : 24,'TAMPINES ST 34' : 466,
                        'YUNG AN RD' : 577,'WOODLANDS AVE 4' : 520,'CHOA CHU KANG NTH 7' : 132,'ANG MO KIO ST 11' : 22,'WOODLANDS AVE 9' : 523,
                        'YUNG LOH RD' : 580,'CHOA CHU KANG DR' : 128,'CHOA CHU KANG ST 54' : 136,'REDHILL LANE' : 386,'KANG CHING RD' : 286,
                        'TAH CHING RD' : 447,'SIMEI ST 5' : 432,'WOODLANDS DR 40' : 529,'WOODLANDS DR 70' : 537,'TAMPINES ST 71' : 473,
                        'WOODLANDS DR 42' : 530,'SERANGOON NTH AVE 3' : 422,'JELAPANG RD' : 229,'BT BATOK ST 22' : 72,'HOUGANG ST 91' : 224,
                        'WOODLANDS AVE 6' : 522,'WOODLANDS CIRCLE' : 524,'CORPORATION DR' : 161,'LOMPANG RD' : 307,'WOODLANDS DR 72' : 539,
                        'CHOA CHU KANG ST 64' : 138,'BT BATOK ST 24' : 73,'JLN TECK WHYE' : 249,'WOODLANDS CRES' : 525,'WOODLANDS DR 60' : 535,
                        'CHANGI VILLAGE RD' : 118,'BT BATOK ST 25' : 74,'HOUGANG AVE 9' : 214,'JURONG WEST CTRL 1' : 264,'WOODLANDS RING RD' : 542,
                        'CHOA CHU KANG AVE 5' : 124,'TOH GUAN RD' : 499,'JURONG WEST ST 61' : 272,'WOODLANDS DR 14' : 527,'HOUGANG ST 92' : 225,
                        'CHOA CHU KANG CRES' : 126,'SEMBAWANG CL' : 399,'CANBERRA RD' : 106,'SEMBAWANG CRES' : 400,'SEMBAWANG VISTA' : 403,
                        'COMPASSVALE WALK' : 160,'RIVERVALE ST' : 390,'WOODLANDS DR 62' : 536,'SEMBAWANG DR' : 401,'WOODLANDS DR 53' : 534,
                        'WOODLANDS DR 52' : 533,'RIVERVALE WALK' : 391,'COMPASSVALE LANE' : 156,'RIVERVALE DR' : 389,'SENJA RD' : 413,
                        'JURONG WEST ST 65' : 275,'RIVERVALE CRES' : 388,'WOODLANDS DR 44' : 531,'COMPASSVALE DR' : 155,'WOODLANDS DR 16' : 528,
                        'COMPASSVALE RD' : 158,'WOODLANDS DR 73' : 540,'HOUGANG ST 31' : 219,'JURONG WEST ST 64' : 274,'WOODLANDS DR 71' : 538,
                        'YISHUN ST 20' : 565,'ADMIRALTY DR' : 0,'COMPASSVALE ST' : 159,'BEDOK RESERVOIR VIEW' : 46,'YUNG SHENG RD' : 582,
                        'ADMIRALTY LINK' : 1,'SENGKANG EAST WAY' : 409,'ANG MO KIO ST 32' : 25,'ANG MO KIO ST 52' : 28,'BOON TIONG RD' : 63,
                        'JURONG WEST ST 62' : 273,'ANCHORVALE LINK' : 10,'CANBERRA LINK' : 105,'COMPASSVALE CRES' : 154,'CLEMENTI ST 12' : 148,
                        'MONTREAL DR' : 336,'WELLINGTON CIRCLE' : 511,'SENGKANG EAST RD' : 408,'JURONG WEST AVE 3' : 262,'ANCHORVALE LANE' : 9,
                        'SENJA LINK' : 412,'EDGEFIELD PLAINS' : 173,'ANCHORVALE DR' : 8,'SEGAR RD' : 396,'FARRER PK RD' : 180,'PUNGGOL FIELD' : 374,
                        'EDGEDALE PLAINS' : 172,'ANCHORVALE RD' : 11,'CANTONMENT CL' : 109,'JLN MEMBINA' : 244,'FERNVALE LANE' : 182,
                        'JURONG WEST ST 25' : 267,'CLEMENTI ST 11' : 147,'PUNGGOL FIELD WALK' : 375,'KLANG LANE' : 301,'PUNGGOL CTRL' : 371,
                        'JELEBU RD' : 230,'BUANGKOK CRES' : 93,'WOODLANDS DR 75' : 541,'BT BATOK WEST AVE 5' : 83,'JELLICOE RD' : 231,
                        'PUNGGOL DR' : 372,'JURONG WEST ST 24' : 266,'SEMBAWANG WAY' : 404,'FERNVALE RD' : 184,'BUANGKOK LINK' : 95,
                        'FERNVALE LINK' : 183,'JLN TIGA' : 252,'YUAN CHING RD' : 576,'COMPASSVALE LINK' : 157,'MARINE PARADE CTRL' : 326,
                        'COMPASSVALE BOW' : 153,'PUNGGOL RD' : 377,'BEDOK CTRL' : 34,'PUNGGOL EAST' : 373,'SENGKANG CTRL' : 406,'TAMPINES CTRL 7' : 456,
                        'SENGKANG WEST AVE' : 410,'PUNGGOL PL' : 376,'CANTONMENT RD' : 110,'GHIM MOH LINK' : 193,'SIMEI LANE' : 427,
                        'YISHUN ST 41' : 569,'TELOK BLANGAH ST 31' : 491,'JLN KAYU' : 240,'LOR 1A TOA PAYOH' : 309,'PUNGGOL WALK' : 378,
                        'SENGKANG WEST WAY' : 411,'BUANGKOK GREEN' : 94,'PUNGGOL WAY' : 379,'YISHUN ST 31' : 568,'TECK WHYE CRES' : 485, 
                        'MONTREAL LINK' : 337,'UPP SERANGOON CRES' : 506,'SUMANG LINK' : 445,'SENGKANG EAST AVE' : 407,'YISHUN AVE 1' : 552,
                        'ANCHORVALE CRES' : 7,'ANCHORVALE ST' : 12,'TAMPINES CTRL 8' : 457,'YISHUN ST 51' : 571,'UPP SERANGOON VIEW' : 508,
                        'TAMPINES AVE 1' : 449,'BEDOK RESERVOIR CRES' : 44,'ANG MO KIO ST 61' : 29,'DAWSON RD' : 164,'FERNVALE ST' : 185,
                        'HOUGANG ST 32' : 220,'TAMPINES ST 86' : 479,'SUMANG WALK' : 446,'CHOA CHU KANG AVE 7' : 125,'KEAT HONG CL' : 287,
                        'JURONG WEST CTRL 3' : 265,'KEAT HONG LINK' : 288,'ALJUNIED AVE 2' : 4,'CANBERRA CRES' : 104,'SUMANG LANE' : 444,
                        'CANBERRA ST' : 107,'ANG MO KIO ST 44' : 26,'ANG MO KIO ST 51' : 27,'BT BATOK EAST AVE 6' : 69,'BT BATOK WEST AVE 9' : 87,
                        'CANBERRA WALK' : 108,'WOODLANDS RISE' : 543,'TAMPINES ST 61' : 472,'YISHUN ST 43' : 570
                        }
    return street_name_dict[street_name]
    

if st.button('Predict'):
    try:
        input_data = np.array([
            int(years), int(months), town_mapping(town), flat_type_mapping(flat_type),
            block, street_name_mapping(street_name), int(flr_area_sqm), flat_model_mapping(flat_model),
            int(stry_start), int(stry_end), int(re_les_year), int(re_les_month), int(les_coms_dt)
        ]).reshape(1, -1)

        # Scaling
        scaling = scaler_flat.transform(input_data)
        # Prediction
        prediction = xgb_model.predict(scaling)
        st.success(f"## :green[*The predicted resale flat price is:*] $ {prediction[0]:.2f} ")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        st.error(f"Error: {e}")
