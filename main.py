'''
Author: LXJ
Date: 2021-03-28 22:21:19
LastEditTime: 2021-06-22 11:46:38
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \python_opencv\camera_test.py
'''
from EnergyMac import *
from CreateTrackerBar import *
from GetFrame import *
from getInform import *
import numpy as np
import time
import cv2


jsonFilePath = "C:/Users/LXJ/Desktop/fanwheel/debug_settings.json"
with open(jsonFilePath,'r',encoding = 'utf-8') as load_f:
    load_dict = json.load(load_f,strict=False)
    
if __name__ == '__main__':
    jsonFilePath = "C:/Users/LXJ/Desktop/fanwheel/debug_settings.json"

    useCamera, useVideo, videoPath, \
    frameWidth, frameHeight, exposuretime, showOriC, \
    videoSpeed, showOriV, \
    preDebug, rsignDebug, flowbarDebug, armorDebug, smallfanDebug, \
    lowHblue, lowSblue, lowVblue, highHblue, highSblue, highVblue, \
    lowHred, lowSred, lowVred, highHred, highSred, highVred, \
    gblurx, gblury, \
    rsDilateX, rsDilateY, MaxRsS, MinRsS, MaxRsRatio, \
    Rin, Rout, MaxlongSide, MinlongSide, fbMinEDdis, \
    rinRatio, sideRatio, \
    MaxdeltaAngle, MindeltaAngle = getInform(jsonFilePath)

    hsvParaBlue = np.array([lowHblue, lowSblue, lowVblue, highHblue, highSblue, highVblue])
    hsvParaRed = np.array([lowHred, lowSred, lowVred, highHred, highSred, highVred])   
    GBSize = [gblurx, gblury]
    RsDilateSize = [rsDilateX, rsDilateY]


    DebugInit(preDebug, rsignDebug, flowbarDebug, armorDebug, smallfanDebug, \
            [hsvParaBlue, GBSize], \
            [RsDilateSize, MaxRsS, MinRsS, MaxRsRatio], \
            [Rin, Rout, MaxlongSide, MinlongSide, fbMinEDdis], \
            [rinRatio, sideRatio], \
            [MaxdeltaAngle, MindeltaAngle])

    if useCamera:
        camera = GetFrame()
        em = EnergyMac(frameWidth, 
                        frameHeight, 
                        hsvParaBlue, 
                        GBSize,
                        RsDilateSize,
                        MaxRsS, 
                        MinRsS, 
                        MaxRsRatio,
                        Rin, 
                        Rout, 
                        MaxlongSide, 
                        MinlongSide, 
                        fbMinEDdis,
                        rinRatio, 
                        sideRatio,
                        MaxdeltaAngle, 
                        MindeltaAngle,
                        hsvParaRed)

        camera.StartCamera()
        camera.SetCamera(frameWidth, frameHeight, exposuretime)

        em.SwitchDebug(preDebug, rsignDebug, flowbarDebug, armorDebug, smallfanDebug)

        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter('D:/python_file/fanwheel.avi', fourcc, 50.0, (1080, 640))


        while True:
            frame = camera.GetOneFrame()
            #out.write(frame)
            #x, y, z = em.GetHitPoint(frame)
            #print(x, y ,z)
            UpdateEMPara(preDebug, rsignDebug, flowbarDebug, armorDebug, smallfanDebug, em)
            if showOriC:
                cv2.imshow("ori", frame)
            key = cv2.waitKey(1)
            if key == 27:
                camera.EndCamera()
                break

    else:
        if useVideo:
            capture = cv2.VideoCapture(videoPath)
            #ret, frame = capture.read()
            #prevframe = frame
            if not capture.isOpened():
                print("Can't open video")
                exit()

            em = EnergyMac(capture.get(3),
                            capture.get(4), 
                            hsvParaBlue, 
                            GBSize,
                            RsDilateSize,
                            MaxRsS, 
                            MinRsS, 
                            MaxRsRatio,
                            Rin, 
                            Rout, 
                            MaxlongSide, 
                            MinlongSide, 
                            fbMinEDdis,
                            rinRatio, 
                            sideRatio,
                            MaxdeltaAngle, 
                            MindeltaAngle,
                            hsvParaRed)
            
            em.SwitchDebug(preDebug, rsignDebug, flowbarDebug, armorDebug, smallfanDebug)

            t0 = time.time()
            fps = 0
            while True:
                ret, frame = capture.read()
                #nextframe = frame
                frame = cv2.flip(frame, 1)
                if not ret:
                    print("Can't get frame")
                    break

                #diff = cv2.absdiff(prevframe, nextframe)
                #print(diff)
                #prevframe = nextframe
                #em.GetHitPoint(frame)
                x, y, z = em.GetHitPoint(frame, 1, 1)
                #print(x, y, z)
                UpdateEMPara(preDebug, rsignDebug, flowbarDebug, armorDebug, smallfanDebug, em)
                if showOriV:
                    cv2.imshow("ori", frame)
                key = cv2.waitKey(int(6000/videoSpeed))#500000
                if key == 27:
                    break
                fps += 1
                t1 = time.time()
                if (t1 - t0) > 1:
                    #print(fps)
                    fps = 0
                    t0 = time.time()

            capture.release()
            
cv2.destroyAllWindows()