import cv2
import os
import dlib
import numpy as np
import pdb
import scipy.io as sio
import imutils
#from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier
from skimage import feature
from sklearn.model_selection import GridSearchCV


import frontalize
import camera_calibration as calib
from lpproj import LocalityPreservingProjection 



cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.namedWindow('aligned face',cv2.WINDOW_NORMAL)
predictor_path =os.path.join(os.path.dirname(__file__),'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)




def readImageOnly(directory):
    listOfImages=[]
    labels=[]
    for ind,folder in enumerate(os.listdir(directory)):
        for path in os.listdir(os.path.join(directory,folder)):
            path=os.path.join(directory,folder,path)
            image=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
            listOfImages.append(image)
            labels.append(folder)
            #print(folder)
        #if(ind>2):
        #    break

    print(".......................................... \n")    
    return listOfImages,labels


def readVideosAndLabels(directory):
    namesUnique=os.listdir(directory)
    labels=[]
    faceMatrix=None
    print(namesUnique)

    indiceToSymmetry=[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,  27,26,25,24,23, 22,21,20,19,18, 28,29,30,31, 36,35,35,33,32, 46,45,44,43,48,47,  40,39,38,37,42,41,  55,54,53,52,51,50,49,60,59,58,57,56, 65,64,63,62,61,68,67,66]
    indiceToSymmetry=list(np.array(indiceToSymmetry)-1)

    model3D = frontalize.ThreeD_Model( "./frontalization_models/model3Ddlib.mat", 'model_dlib')

    eyemask = np.asarray(sio.loadmat('frontalization_models/eyemask.mat')['eyemask'])
    for ind,folder in enumerate(namesUnique):
        for path in os.listdir(os.path.join(directory,folder)):
            
            path=os.path.join(directory,folder,path)
            cap=cv2.VideoCapture(path)
            count=0
            trackingLost=True
            print(folder)
            """
            if(folder!="Romuald"):
                print("a")
                
                if(folder=="Stephane"):
                    pass
                else:
                   break
            """
            if(folder!="Florian"):
                continue
            while(True):
                
                ret,imageO=cap.read()
                
                if(count%2==1):
                    
                    if(not ret):
                        break
                    #print(imageO.shape)
                    #image = cv2.resize(image,(700,700))
                    if(folder=="Faddwzi"):
                        #image=imutils.rotate_bound(image, -90)
                        break
                        #pass
                    else:
                        #image=imutils.rotate_bound(image, 90)
                        image=np.ascontiguousarray(np.transpose(np.copy(imageO), (1, 0, 2)), dtype=np.uint8)
                        #image=np.ascontiguousarray(rotateImage(imageO),dtype=np.uint8)
                    #image=imutils.rotate_bound(image, 90)
                    #image=imutils.resize(image, width=370)
                    
                    #imageWithLandmarks,landmarks,_=searchForFaceInTheWholeImage(np.copy(image))
                    if(trackingLost):
                        landmarks,faceROI,trackingLost,image=trackFaceInANeighborhoodAndDetectLandmarks(np.copy(image),faceROI=[0, 0,image.shape[0]-1, image.shape[1]-1],drawBoundingBoxes=True)
                    else:
                        landmarks,faceROI,trackingLost,image=trackFaceInANeighborhoodAndDetectLandmarks(np.copy(image),faceROI[0],drawBoundingBoxes=True)
                    #print(trackingLost,faceROI)
                    cv2.imshow('image',image)
                    cv2.waitKey(1)
                    if(trackingLost):
                        continue
                    #pdb.set_trace()
                    #landmarks[0][:,1]=landmarks[0][:,1]+np.sign(image.shape[]landmarks[0][:,1])
                    landmarks[0][:,[0, 1]] = landmarks[0][:,[1, 0]]
                    #landmarks[0][range(0,68)]=landmarks[0][indiceToSymmetry,:]
                    #pdb.set_trace()
                    #######registeredFace=performFaceAlignment(image,landmarks[0],cols=600,rows=600)
                    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, landmarks[0])
                    # load mask to exclude eyes from symmetry
                    # perform frontalization
                    frontal_raw, registeredFace = frontalize.frontalize(imageO, proj_matrix, model3D.ref_U, eyemask)
                    
                    if(registeredFace is not None):
                        
                        faceVector=((registeredFace)/255.0).reshape(-1)
                        if(faceMatrix is None):
                            faceMatrix=faceVector
                        else:
                            faceMatrix=np.vstack((faceMatrix,faceVector))
                        labels.append(folder)
                        for landmark in landmarks:
                            cv2.polylines(image,np.int32(landmark.reshape((-1,1,2))),True,(0,0,255),3)
                        cv2.imshow('image333',frontal_raw)
                        cv2.imshow('aligned face',registeredFace)
                        cv2.waitKey(1)
                    cv2.imwrite(os.path.join(directory,folder,str(count)+'.jpg'), registeredFace)  

                count=count+1
            cap.release()
    return faceMatrix,labels 

def readRectifiedImages(directory):
    try:
        matFile=sio.loadmat(os.path.join(directory,"train.mat"))
        faceMatrix=matFile["images"]
        labels=[s.strip() for s in matFile["personID"]]

    except:
        namesUnique=os.listdir(directory)
        labels=[]
        faceMatrix=None
        print(namesUnique)
        print("alopre")
        lbp=LocalBinaryPatterns(numPoints=8, radius=1)

        for ind,folder in enumerate(namesUnique):
            for path in os.listdir(os.path.join(directory,folder)):
                path=os.path.join(directory,folder,path)
                #image=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)
                image=cv2.imread(path)
                image = imutils.resize(image, height=100)
                #image=image.reshape(-1)
                faceVector=extractDataForOneImage(image,lbp,8,8).reshape(-1)
                #faceVector=((image-np.mean(image))/np.std(image)).reshape(-1)
                if(faceMatrix is None):
                    faceMatrix=faceVector
                    labels.append(folder)
                else:
                    faceMatrix=np.vstack((faceMatrix,faceVector))
                    labels.append(folder)
                    cv2.imshow('ireremage',image)
                    cv2.waitKey(1)
        sio.savemat(os.path.join(directory,"train.mat"),{"images":faceMatrix,"personID":labels})
    print("alo")
    return faceMatrix,labels     





def performFaceRecognitionWithFrontalisation(image,  recognizer,names, model3D, eyemask):
    #indiceToSymmetry=[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,  27,26,25,24,23, 22,21,20,19,18, 28,29,30,31, 36,35,35,33,32, 46,45,44,43,48,47,  40,39,38,37,42,41,  55,54,53,52,51,50,49,60,59,58,57,56, 65,64,63,62,61,68,67,66]
    #indiceToSymmetry=list(np.array(indiceToSymmetry)-1)


        
    imageWithLandmarks,landmarks,faceROIs=searchForFaceInTheWholeImage(np.copy(image))
    result=[]
    for landmark,faceROI in zip(landmarks,faceROIs):
        #registeredFace=performFaceAlignment(image,landmark,cols=600,rows=600)
        #landmark[range(0,68),:]=landmark[indiceToSymmetry,:]

        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, landmark)
        # load mask to exclude eyes from symmetry
        # perform frontalization
        frontal_raw, registeredFace = frontalize.frontalize(image, proj_matrix, model3D.ref_U, eyemask)
        registeredFace = imutils.resize(registeredFace, height=100)
          
        cv2.imshow('el3ab',cv2.cvtColor(registeredFace, cv2.COLOR_BGR2GRAY))

        cv2.waitKey(1)
        
        pred,conf=recognizer.predict(cv2.cvtColor(registeredFace,cv2.COLOR_BGR2GRAY))
        
        
        xdir=2*np.array([-49.6694, -0.3201, 1.0163])
        ydir=4*np.array([-0.9852,-3.1128,15.0628])
        zdir=-np.array([-1.658,747.159,154.29])/5.0
        origin=np.array([-0.0845, -74.7281, 27.2774])
        image,_=model3D.drawCoordinateSystems(np.hstack((rmat,tvec)),image,_3Dpoints=np.array([origin,origin+xdir,origin+ydir,origin+zdir]))
        image=model3D.drawCandideMesh(np.hstack((rmat,tvec)),image)
        # 
        #distance, indice = svm.kneighbors(faceVector.reshape(1,-1))
        #indice=classifier.predict(embeddedFace.reshape(1,-1))
        #pred=svm_Tuned.predict(faceVector.reshape(1,-1))[0]
        
        ###distance, indice = nbrs.kneighbors(embeddedFace.reshape(1,-1))
        #indice=classifier.predict(embeddedFace.reshape(1,-1))
        identity=list(names.keys())[list(names.values()).index(pred)]
        print(pred,conf)
        ##(#cv2.rectangle(image,(faceROI[1],faceROI[0]),(faceROI[1]+faceROI[3],faceROI[0]+faceROI[2]),(255,0,255),2)      
        ###cv2.putText(image,indice[0],(faceROI[1],faceROI[0]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        cv2.rectangle(image,(faceROI[1],faceROI[0]),(faceROI[1]+faceROI[3],faceROI[0]+faceROI[2]),(255,0,255),2)
        
        cv2.putText(image,identity ,(faceROI[1],faceROI[0]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),thickness=2)
        ###result.append(indice[0])
        #print (distance,labels[indice[0][0]])
        """if(distance<threshold):
            
            cv2.rectangle(image,(faceROI[1],faceROI[0]),(faceROI[1]+faceROI[3],faceROI[0]+faceROI[2]),(255,0,255),2)      
            cv2.putText(image,labels[indice[0][0]]   ,(faceROI[1],faceROI[0]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
            
            result.append(labels[indice[0][0]])
        
        else:
            cv2.rectangle(image,(faceROI[1],faceROI[0]),(faceROI[1]+faceROI[3],faceROI[0]+faceROI[2]),(255,0,255),2)      
            cv2.putText(image,"unknown"   ,(faceROI[1],faceROI[0]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
            
            result.append("unknown")
        """
        
    cv2.imshow('image',image)
    cv2.waitKey(1)
    return result,faceROIs

   

def getThirdPointOnEquelateralTriangle(point1,point2):
    a=((point1-point2)[1])/(point2-point1)[0]
    dist=np.linalg.norm(point1-point2)**2
    x3 = np.sqrt(((np.sqrt(3)/2)*dist)/(a**2+1))+((point1+point2)/2)[1]
    y3= a*np.sqrt(((np.sqrt(3)/2)*dist)/(a**2+1))+((point1+point2)/2)[0]
    return np.array([y3,x3])

def performFaceAlignment(frame,landmarks,cols=600,rows=600):
   
    l=420-180
    size=(420+np.int(0.6*l)-180+np.int(0.6*l),np.int(2.2*l))
    faceROI=[0 ,0, frame.shape[0], frame.shape[1]]
    N=70
    eyePositions=[]     
    leftEye=(landmarks[36,:]+landmarks[39,:])/2
    rightEye=(landmarks[42,:]+landmarks[45,:])/2
    eyePositions.append(np.hstack((leftEye,rightEye)))
    equilateralPoint=getThirdPointOnEquelateralTriangle(leftEye, rightEye)
    inputPts=np.float32(np.array([leftEye,rightEye,equilateralPoint])).reshape(1,3,2).astype(np.int)
    outputPts=np.float32(np.array([[180,200],[420,200],[300,408]])).reshape(1,3,2).astype(np.int)
    
    try:     
        #similarity=cv2.estimateRigidTransform(inputPts,outputPts,False)
        similarity=cv2.estimateAffinePartial2D(inputPts,outputPts)[0]
        
        registeredFace = cv2.warpAffine(frame,similarity,(cols,rows))
        
        cv2.polylines(frame,np.int32(landmarks.reshape((-1,1,2))),True,(255,255,0),3)
        #cv2.polylines(frame,[np.int32(np.array([leftEye,rightEye,equilateralPoint]))],True,(255,255,0),3)
        #cv2.polylines(frame,np.int32(equilateralPoint.reshape((-1,1,2))),True,(255,255,255),7)
        return registeredFace
    except Exception as e: 
        #print(e)
        pass
        return None
"""        
def computeError(y, y_pred):
    return np.sqrt(np.mean((np.sum(y_pred,1)-np.sum(y,1))**2))    
"""    
def computeError(y, y_pred):
    return ((np.sum(np.array(y_pred)==np.array(y))*100.0)/len(y))

    
def trackFaceInANeighborhoodAndDetectLandmarks(image,faceROI,drawBoundingBoxes=True):
       
     N=25
     landmarks=np.zeros([68,2])
     x_top_left=max(faceROI[0]-N,0)
     x_bottom_right=min(image.shape[0],faceROI[0]+faceROI[2]+N)
     y_top_left=max(faceROI[1]-N,0)
     y_bottom_right=min(faceROI[1]+faceROI[3]+N,image.shape[1])
     landmarkList=[]
     boundingBoxList=[]
     if(drawBoundingBoxes):
         #print(y_top_left,x_top_left,y_bottom_right,x_bottom_right,image.shape)
         cv2.rectangle(image,(y_top_left,x_top_left),(y_bottom_right,x_bottom_right),(0,255,0),1)
     ####u can program it better####
     ##faceBoundingBoxes=enumerate(detector(image[x_top_left:x_bottom_right,y_top_left:y_bottom_right],0))
     faceBoundingBoxes=enumerate(detector(cv2.cvtColor(image[x_top_left:x_bottom_right,y_top_left:y_bottom_right,:],cv2.COLOR_BGR2GRAY),0))
     trackingLost=True
     for k,d in faceBoundingBoxes:
         ##shape = predictor(image[max(faceROI[0]-N,0):min(image.shape[0],faceROI[0]+faceROI[2]+N),max(faceROI[1]-N,0):min(faceROI[1]+faceROI[3]+N,image.shape[1])],d)
         shape = predictor(cv2.cvtColor(image[x_top_left:x_bottom_right,y_top_left:y_bottom_right,:],cv2.COLOR_BGR2GRAY),d)
         for i in range(0,68):
             landmarks[i,:]=np.array([shape.part(i).x+y_top_left,shape.part(i).y+x_top_left])
         """for landmark in landmarks:
             cv2.polylines(image,np.int32(landmark.reshape((-1,1,2))),True,(0,0,255),3)"""  
         faceROI=[x_top_left+d.top(), y_top_left+d.left(),d.height(), d.width()]
         
         trackingLost=False
         landmarkList.append(landmarks)
         boundingBoxList.append(faceROI)
         
         if(drawBoundingBoxes):      
             cv2.rectangle(image,(faceROI[1],faceROI[0]),(faceROI[1]+faceROI[3],faceROI[0]+faceROI[2]),(255,0,255),2)             
     return landmarkList,boundingBoxList,trackingLost,image

def searchForFaceInTheWholeImage(image):
           
    faceROI=[0, 0,image.shape[0]-1, image.shape[1]-1]

    landmarks,faceROI,_,image=trackFaceInANeighborhoodAndDetectLandmarks(image,faceROI)
    for landmark in landmarks:
        cv2.polylines(image,np.int32(landmark.reshape((-1,1,2))),True,(0,0,255),3)
    
    try: 
        for i in range(0,landmarks[0].shape[0]):
            cv2.putText(image,str(i),(int(landmarks[0][i,0]),int(landmarks[0][i,1])),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0))
    except Exception as e:
        #print(e)
        pass
    
    cv2.imshow("zdada",image)
    cv2.waitKey(1)
    
    return image,landmarks,faceROI
    

def predictOnTestImages(path,recognizer,personDict):

    
    faceArray,labels= readImageOnly(path)
    labelNumPerson=[personDict.get(n, n) for n in labels]
    predictions=[]
    for img in faceArray:
        pred,conf=recognizer.predict(img)
        predictions.append(pred)
        #print(conf)
    #pdb.set_trace()
    print("Test accuracy: ",computeError(labelNumPerson,predictions))
    return


if __name__ == "__main__":
    
    print("reading images step started")
    imgArray,labels=readImageOnly(".//Appstud-Frontalization-train")
    ##imgArray,labels=readImageOnly(".//frontal_appstud_train")
    #imgArray,labels=readImageOnly(".//pose_appstud_train")

    #faceMatrix,labels=readRectifiedImages(".//Appstud-Frontalization-train")
    print("reading images step finished")
    
    names=(list(set(labels)))    
    labelNumPersonDict = { name : i for i,name in  enumerate(names) }
    labelNumPerson=[labelNumPersonDict.get(n, n) for n in labels]
    
    
   
    recognizer = cv2.face.LBPHFaceRecognizer_create(1,8,8,8)
    recognizer.train( imgArray, np.array(labelNumPerson))
    
    predictOnTestImages(".//Frontalization_data_train",recognizer, labelNumPersonDict)
    ##predictOnTestImages(".//frontal_appstud_test",recognizer, labelNumPersonDict)
    ##predictOnTestImages(".//frontal_appstud_test",recognizer, labelNumPersonDict)


    cap = cv2.VideoCapture(0)
    model3D = frontalize.ThreeD_Model( "./frontalization_models/model3Ddlib.mat", 'model_dlib')

    eyeMask = np.asarray(sio.loadmat('frontalization_models/eyemask.mat')['eyemask'])
    model3D.out_A=np.asmatrix(np.array([[572.216,0,316.76],[0,577.67,223.65],[0,0,1]]), dtype='float32') #3x3
    model3D.distCoeff=np.array([ 0.16975413 ,-0.39957115, -0.01705758 , 0.00303078, -0.25477128])
    #print(labelNumPersonDict)
    while(True):
        ret, frame = cap.read()
        

        if(ret):
            person=performFaceRecognitionWithFrontalisation(frame,  recognizer, labelNumPersonDict, model3D, eyeMask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    """
    images= readImageOnly(".//lfw")
    person=[ performFaceRecognition(image,mean,eigenVectors,databaseOfEmbeddedFaces,labels,threshold=10) for image in images ]
    """
       
   
    
