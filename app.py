import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#######################################################
import uuid ## random id generator
from streamlit_option_menu import option_menu
import streamlit as st
import os
import shutil
import cv2
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
from test import test
import torch
import datetime
import hashlib
#######################################################



ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VISITOR_DB = os.path.join(ROOT_DIR, "visitor_database")
VISITOR_HISTORY = os.path.join(ROOT_DIR, "visitor_history")
COLOR_DARK  = (255,255,255)
COLOR_WHITE = (75,110,192)
COLS_INFO   = ['Name']
COLS_ENCODE = [f'v{i}' for i in range(512)]
sidebar_color = (236,239,244)
## Database
data_path       = VISITOR_DB
file_db         = 'visitors_db.csv'         ## To store user information
file_history    = 'visitors_history.csv'    ## To store visitor history information

## Image formats allowed
allowed_image_type = ['.png', 'jpg', '.jpeg']


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def initialize_data():
    if os.path.exists(os.path.join(data_path, file_db)):
        # st.info('Database Found!')
        df = pd.read_csv(os.path.join(data_path, file_db))

    else:
        # st.info('Database Not Found!')
        df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
        df.to_csv(os.path.join(data_path, file_db), index=False)

    return df



def add_data_db(df_visitor_details):
    try:
        df_all = pd.read_csv(os.path.join(data_path, file_db))

        if not df_all.empty:
            df_all = pd.concat([df_all,df_visitor_details], ignore_index=False)
            df_all.drop_duplicates(keep='first', inplace=True)
            df_all.reset_index(inplace=True, drop=True)
            df_all.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Details Added Successfully!')
        else:
            df_visitor_details.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Initiated Data Successfully!')

    except Exception as e:
        st.error(e)

def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

def attendance(id, name):
    f_p = os.path.join(VISITOR_HISTORY, file_history)
    # st.write(f_p)

    now = datetime.datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    df_attendace_temp = pd.DataFrame(data={ "id"            : [id],
                                            "visitor_name"  : [name],
                                            "Timing"        : [dtString]
                                            })

    if not os.path.isfile(f_p):
        df_attendace_temp.to_csv(f_p, index=False)
        # st.write(df_attendace_temp)
    else:
        df_attendace = pd.read_csv(f_p)
        df_attendace = pd.concat([df_attendace,df_attendace_temp])
        df_attendace.to_csv(f_p, index=False)

def view_attendace():
    f_p = os.path.join(VISITOR_HISTORY, file_history)
    # st.write(f_p)
    df_attendace_temp = pd.DataFrame(columns=["id",
                                              "visitor_name", "Timing"])

    if not os.path.isfile(f_p):
        df_attendace_temp.to_csv(f_p, index=False)
    else:
        df_attendace_temp = pd.read_csv(f_p)

    df_attendace = df_attendace_temp.sort_values(by='Timing',
                                                 ascending=False)
    df_attendace.reset_index(inplace=True, drop=True)

    st.write(df_attendace)

    if df_attendace.shape[0]>0:
        id_chk  = df_attendace.loc[0, 'id']
        id_name = df_attendace.loc[0, 'visitor_name']

        selected_img = st.selectbox('Search Image using ID',
                                    options=['None']+list(df_attendace['id']))

        avail_files = [file for file in list(os.listdir(VISITOR_HISTORY))
                       if ((file.endswith(tuple(allowed_image_type))) &
                                                                              (file.startswith(selected_img) == True))]

        if len(avail_files)>0:
            selected_img_path = os.path.join(VISITOR_HISTORY,
                                             avail_files[0])
            #st.write(selected_img_path)

            ## Displaying Image
            st.image(Image.open(selected_img_path))

def crop_image_with_ratio(img, height,width,middle):
    h, w = img.shape[:2]
    h=h-h%4
    new_w = int(h / height)*width
    startx = middle - new_w //2
    endx=middle+new_w //2
    if startx<=0:
        cropped_img = img[0:h, 0:new_w]
    elif endx>=w:
        cropped_img = img[0:h, w-new_w:w]
    else:
        cropped_img = img[0:h, startx:endx]
    return cropped_img

################################################### Defining Static Data ###############################################

user_color      = '#756c83'
h1_color        = '#b9e1dc'
text_color      = '#252630'
title_webapp    = "FORTIFY"

html_temp = f"""
            <div style="background-color:{user_color};padding:12px">
            <h1 style="color:{h1_color};text-align:center;font-size: 38px;">{title_webapp}</h1>
            </div>
            """
st.markdown(html_temp, unsafe_allow_html=True)




###################### Defining Static Paths ###################4


if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)

if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)
# st.write(VISITOR_HISTORY)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device,keep_all=True
        )
########################################################################################################################


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-color: #756c83;
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}}
[data-testid ="stHeader"] {{
    background-color: #ECEFF4;         
}}
body{{
    color: #252630;
}}
a{{
    color: #000000;               
}}

</style>
"""


#83AFBD

st.markdown(page_bg_img, unsafe_allow_html=True)

COLS_INFO = ['Name', 'Password'] 

def main():
    st.sidebar.image("F.png", width=250)

    with st.sidebar:
        option = option_menu(
            "Navigation",
            ["Home", "Add to Database", "Visitor Validation", "View Visitor History", ],
            icons=["house", "person-plus", "person-check", "book"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            key="main_navigation"
        )

    
    if option == "Home":
        # Welcoming Header
        
    # Welcoming Header with Logo
        st.markdown("""
    <div style="background-color:#756c83;padding:10px;border-radius:10px;display:flex;justify-content:space-between;align-items:center;">
        <h1 style="color:#fefefe;text-align:center;margin:auto 0;">Welcome to Biometric Login and Verification System!</h1>
        <img src= "icon.png" style="height:80px;margin-left:20px;">
    </div>
    """, unsafe_allow_html=True)

    # Problem Statement
        st.markdown("""
    <h3 style="color:#fbfbfb;">
        Secure Your Access with Two-Factor Authentication
    </h3>
    <p style="color:#fbfbfb;">
        Welcome to our state-of-the-art two-factor authentication platform, where we use advanced facial recognition technology to enhance your login security. Our system is designed to provide seamless and secure access, combining the convenience of facial recognition with robust anti-spoofing measures.
    </p>
    <h3 style="color:#fbfbfb;">
        Why Choose Our Platform?
    </h3>
    <p style="color:#fbfbfb;">
        - Advanced Facial Recognition: Our cutting-edge facial recognition technology ensures quick and accurate identification.<br>
        - Anti-Spoofing Measures: We employ sophisticated anti-spoofing techniques to protect against fraudulent access attempts.<br>
        - Enhanced Security: By integrating facial recognition with two-factor authentication, we provide an extra layer of security for your accounts and transactions.
    </p>
    <p style="color:#fbfbfb;">
        Explore our website to learn more about how our platform safeguards your information and provides a secure, user-friendly authentication experience.
    </p>
    """, unsafe_allow_html=True)

    # Features Section with Icons or Images
        st.markdown("""
    <h3 style="color:#fbfbfb;">
        Key Features
    </h3>
    <div style="display:flex; justify-content:space-around; margin:20px 0;">
        <div style="text-align:center;">
            <img src="{image_path}" style="border-radius:50%;">
            <h3 style="color:#fbfbfb;">Scalability</h3>
            <p style="color:#fbfbfb;">Our system is designed to handle a large number of users seamlessly.</p>
        </div>
        <div style="text-align:center;">
            <img src="/Users/gauridubey/Desktop/what/shreya/face-recognition-attendance-anti-spoofing/cat.jpg" style="border-radius:50%;">
            <h3 style="color:#fbfbfb;">Relevance</h3>
            <p style="color:#fbfbfb;">Applicable across various industries including finance, education, and corporate sectors.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample Image or Video
        st.markdown("""
    <h3 style="color:#fbfbfb;">
        See it in Action!
    </h3>
    """, unsafe_allow_html=True)
        sample_image = "/Users/gauridubey/Desktop/what/shreya/face-recognition-attendance-anti-spoofing/h.jpg"  # Replace with the actual path to sample video
        st.image(sample_image, caption="Face Recognition in Action", use_column_width=True)

    # Interactive Components
        st.markdown("""<h3 style="color:#fbfbfb;"> Try it Out</h3>""", unsafe_allow_html=True)
        if st.button("Get Started"):
            st.experimental_set_query_params(page="Add to Database")
            st.experimental_rerun()

    # Links to Documentation or External Resources
        st.markdown("""<h3 style="color:#fbfbfb;">Learn More</h3>""", unsafe_allow_html=True)
        st.markdown("""
    <ul style="list-style-type:square;">
        <li style="color:#83afbd;"> <a href="https://google.com" target="_blank" style="color:#83afbd;text-decoration:none;">Project Documentation</a></li>
        <li style="color:#83afbd;"> <a href="https://github.com" target="_blank" style="color:#83afbd;text-decoration:none;">GitHub Repository</a></li>
        <li style="color:#83afbd;"> <a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.youtube.com/watch%3Fv%3DdQw4w9WgXcQ&ved=2ahUKEwirkpDq9NGGAxWBX_EDHVdZAuwQ78AJegQIGxAB&usg=AOvVaw0aHtehaphMhOCAkCydRLZU" target="_blank" style="color:#83afbd;text-decoration:none;">Contact Us</a></li>
    </ul>
    """, unsafe_allow_html=True)




    elif option == 'Visitor Validation':
        visitor_id = uuid.uuid1()

        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()

            image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            image_array_copy = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with open(os.path.join(VISITOR_HISTORY, f'{visitor_id}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())
                st.success('Image Saved Successfully!')

                max_faces = 0
                rois = []
                aligned = []
                spoofs = []
                can = []
                face_locations, prob = mtcnn(image_array, return_prob=True)
                boxes, _ = mtcnn.detect(image_array)
                boxes_int = boxes.astype(int)

                if face_locations is not None:
                    for idx, (left, top, right, bottom) in enumerate(boxes_int):
                        img = crop_image_with_ratio(image_array, 4, 3, (left + right) // 2)
                        spoof = test(img, "./resources/anti_spoof_models", device)
                        if spoof <= 1:
                            spoofs.append("REAL")
                            can.append(idx)
                        else:
                            spoofs.append("FAKE")
                    print(can)

                for idx, (left, top, right, bottom) in enumerate(boxes_int):
                    rois.append(image_array[top:bottom, left:right].copy())
                    cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
                    cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(image_array, f"#{idx} {spoofs[idx]}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

                st.image(BGR_to_RGB(image_array), width=720)

                max_faces = len(boxes_int)

                if max_faces > 0:
                    col1, col2 = st.columns(2)
                    face_idxs = col1.multiselect("Select face#", can, default=can)
                    similarity_threshold = col2.slider('Select Threshold for Similarity', min_value=0.0, max_value=3.0, value=0.8)
                    flag_show = False

                    if ((col1.checkbox('Click to proceed!')) & (len(face_idxs) > 0)):
                        dataframe_new = pd.DataFrame()
                        for idx, loc in enumerate(face_locations):
                            torch_loc = torch.stack([loc]).to(device)
                            encodesCurFrame = resnet(torch_loc).detach().cpu()
                            aligned.append(encodesCurFrame)

                        for face_idx in face_idxs:
                            database_data = initialize_data()

                            face_encodings = database_data[COLS_ENCODE].values
                            dataframe = database_data[COLS_INFO + ['Password']]

                            if len(aligned) < 1:
                                st.error(f'Please Try Again for face#{face_idx}!')
                            else:
                                face_to_compare = aligned[face_idx].numpy()
                                dataframe['similarity'] = [np.linalg.norm(e1 - face_to_compare) for e1 in face_encodings]
                                dataframe['similarity'] = dataframe['similarity'].astype(float)

                                dataframe_new = dataframe.drop_duplicates(keep='first')
                                dataframe_new.reset_index(drop=True, inplace=True)
                                dataframe_new.sort_values(by="similarity", ascending=True, inplace=True)
                                dataframe_new = dataframe_new[dataframe_new['similarity'] < similarity_threshold].head(1)
                                dataframe_new.reset_index(drop=True, inplace=True)

                                if dataframe_new.shape[0] > 0:
                                    (left, top, right, bottom) = (boxes_int[face_idx])

                                    rois.append(image_array_copy[top:bottom, left:right].copy())
                                    cv2.rectangle(image_array_copy, (left, top), (right, bottom), COLOR_DARK, 2)
                                    cv2.rectangle(image_array_copy, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(image_array_copy, f"#{dataframe_new.loc[0, 'Name']}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

                                    name_visitor = dataframe_new.loc[0, 'Name']
                                    st.image(BGR_to_RGB(image_array_copy), width=720)
                                    entered_password = st.text_input("Enter your password", type="password")
                                    if entered_password:
                                        hashed_entered_password = hash_password(entered_password)
                                        stored_hashed_password = dataframe_new.loc[0, 'Password'].values[0]
                                        if st.button("Proceed"):
                                            if hashed_entered_password == stored_hashed_password:
                                                attendance(visitor_id, name_visitor)
                                                st.write("Login Successful!")
                                        else:
                                            st.write("Login Unsuccessful! Wrong Password")
                                            
                                else:
                                    st.error(f'No Match Found for the given Similarity Threshold! for face#{face_idx}')
                                    st.info('Please Update the database for a new person or click again!')
                                    attendance(visitor_id, 'Unknown')

                        if flag_show == True:
                            st.image(BGR_to_RGB(image_array_copy), width=720)
                else:
                    st.error('No human face detected.')

        

    elif option == 'View Visitor History':
        view_attendace()

        if st.button('Clear all data'):
            if os.path.exists(os.path.join(VISITOR_HISTORY, file_history)):
                os.remove(os.path.join(VISITOR_HISTORY, file_history))
                st.success('History cleared successfully!')
            else:
                st.info('No history to clear.')

    elif option == 'Add to Database':
        col1, col2, col3 = st.columns(3)

        face_name = col1.text_input('Name:', '')
        face_password = col2.text_input('Password:', type='password')
        pic_option = col2.radio('Upload Picture', options=["Upload a Picture", "Take a Picture with Cam"], key="pic_option")

        if pic_option == 'Upload a Picture':
            img_file_buffer = col3.file_uploader('Upload a Picture', type=allowed_image_type)
            if img_file_buffer is not None:
                file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)

        elif pic_option == 'Take a Picture with Cam':
            img_file_buffer = col3.camera_input("Take a Picture with Cam")
            if img_file_buffer is not None:
                file_bytes = np.frombuffer(img_file_buffer.getvalue(), np.uint8)

        if ((img_file_buffer is not None) & (len(face_name) > 1) & (len(face_password) > 1) & st.button('Click to Save!')):
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with open(os.path.join(VISITOR_DB, f'{face_name}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())

            face_locations, prob = mtcnn(image_array, return_prob=True)
            torch_loc = torch.stack([face_locations[0]]).to(device)
            encodesCurFrame = resnet(torch_loc).detach().cpu()

            df_new = pd.DataFrame(data=encodesCurFrame, columns=COLS_ENCODE)
            df_new['Name'] = face_name
            df_new['Password'] = hash_password(face_password)
            df_new = df_new[COLS_INFO + COLS_ENCODE].copy()

            DB = initialize_data()
            add_data_db(df_new)

if __name__ == "__main__":
    main()
