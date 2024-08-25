import numpy as np
import pandas as pd
import cv2
import redis

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

hostname='redis-11302.c326.us-east-1-3.ec2.cloud.redislabs.com'
port=11302
password='xeP8W9q5FU2jHJf22odW3g5sszFvuz6R'

r=redis.StrictRedis(host=hostname,port=port,password=password)

faceapp=FaceAnalysis(name='buffalo_l',root='insightface_model')
faceapp.prepare(ctx_id=0,det_size=(640,640),det_thresh=0.5)

def search(df,feature_col,test_vector,threshold=0.5):
    df=df.copy()
    x_list=df[feature_col].to_list()
    x=np.asarray(x_list)
    
    similar=pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr=np.array(similar).flatten()
    df['cosine']=similar_arr
    
    data_filter=df.query(f'cosine >= {threshold}')
    if len(data_filter)>0:
        data_filter.reset_index(drop=True,inplace=True)
        argmax=data_filter['cosine'].argmax()
        person_name,person_role=data_filter.loc[argmax][['Name','Role']]
        
    else:
        person_name='Unknown'
        person_role='Unknown'

    return person_name,person_role

import pyttsx3

def speak_name(name):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

    # Speak the provided name
    engine.say(f"{name}, identified!")

    # Wait for the speech to finish
    engine.runAndWait()

def face_prediction(test_image,df,feature_col,threshold=0.5):
    results=faceapp.get(test_image)
    test_copy=test_image.copy()
    for r in results:
        x1,y1,x2,y2=r['bbox'].astype('int')
        embeddings=r['embedding']
        person_name,person_role=search(df,feature_col,test_vector=embeddings,threshold=threshold)

        if person_name=='Unknown':
            color=(0,0,255)
        else:
            color=(0,255,0)
        cv2.rectangle(test_copy,(x1,y1),(x2,y2),color) 
        cv2.putText(test_copy,person_name,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
        
    return test_copy


        
