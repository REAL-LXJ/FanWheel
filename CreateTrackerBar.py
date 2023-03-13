'''
Author: LXJ
Date: 2021-03-29 11:42:21
LastEditTime: 2021-04-24 23:40:32
LastEditors: LXJ
Description: the file is used to create trackerbar for debugging
FilePath: \python_opencv\camera_test\CreateTrackerBar.py
'''
import cv2
import numpy as np

#滑动条参数改变默认处理函数
def nothing(*arg):
    pass

def CreatePretreatmentTrackerBar(hsvPara, GBSize):
    cv2.namedWindow('colorTest')
    # Lower range colour sliders.
    cv2.createTrackbar('lowHue', 'colorTest', hsvPara[0], 255, nothing)
    cv2.createTrackbar('lowSat', 'colorTest', hsvPara[1], 255, nothing)
    cv2.createTrackbar('lowVal', 'colorTest', hsvPara[2], 255, nothing)
    # Higher range colour sliders.
    cv2.createTrackbar('highHue', 'colorTest', hsvPara[3], 255, nothing)
    cv2.createTrackbar('highSat', 'colorTest', hsvPara[4], 255, nothing)
    cv2.createTrackbar('highVal', 'colorTest', hsvPara[5], 255, nothing)
    cv2.createTrackbar('GBx', 'colorTest', GBSize[0], 15, nothing)
    cv2.createTrackbar('GBy', 'colorTest', GBSize[1], 15, nothing)

def CreateRsignTrackerBar(RsDilatePara, MaxRsS, MinRsS, MaxRsRatio):
    cv2.namedWindow('RsTest')
    cv2.createTrackbar('RsDilateX', 'RsTest', RsDilatePara[0], 15, nothing)
    cv2.createTrackbar('RsDilateY', 'RsTest', RsDilatePara[1], 15, nothing)
    cv2.createTrackbar('MaxRsS', 'RsTest', MaxRsS, 1000, nothing)
    cv2.createTrackbar('MinRsS', 'RsTest', MinRsS, 500, nothing)
    cv2.createTrackbar('MaxRsRatio', 'RsTest', int(MaxRsRatio*50), 100, nothing)

def CreateFlowBarTrackerBar(Rin, Rout, MaxlongSide, MinlongSide, fbMinEDdis):
    cv2.namedWindow('FBTest')
    cv2.createTrackbar('Rin', 'FBTest', Rin, 200, nothing)
    cv2.createTrackbar('Rout', 'FBTest', Rout, 200, nothing)
    cv2.createTrackbar('MaxlongSide', 'FBTest', int(MaxlongSide*10), 1000, nothing)
    cv2.createTrackbar('MinlongSide', 'FBTest', int(MinlongSide*10), 1000, nothing)
    cv2.createTrackbar('fbMinEDdis', 'FBTest', fbMinEDdis, 200, nothing)

def CreateArmorTrackerBar(rinRatio, sideRatio):
    cv2.namedWindow('ArTest')
    cv2.createTrackbar('rinRatio', 'ArTest', int(rinRatio*20), 100, nothing)
    cv2.createTrackbar('sideRatio', 'ArTest', int(sideRatio*100), 100, nothing)

def CreateSmallfanTrackbar(MaxdeltaAngle, MindeltaAngle):
    cv2.namedWindow('SfanTest')
    cv2.createTrackbar('MaxdeltaAngle', 'SfanTest', MaxdeltaAngle, 100, nothing)
    cv2.createTrackbar('MindeltaAngle', 'SfanTest', MindeltaAngle, 100, nothing)

def UpdateEMPara(preDebug, rsignDebug, flowbarDebug, armorDebug, smallfanDebug, em): 
    if preDebug:
        lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
        lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
        lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
        highHue = cv2.getTrackbarPos('highHue', 'colorTest')
        highSat = cv2.getTrackbarPos('highSat', 'colorTest')
        highVal = cv2.getTrackbarPos('highVal', 'colorTest')
        GBx = cv2.getTrackbarPos('GBx', 'colorTest')
        if GBx % 2 == 0:
            GBx -= 1
        GBy = cv2.getTrackbarPos('GBy', 'colorTest')
        if GBy % 2 == 0:
            GBy -= 1
        em.SetPretreatmentPara([lowHue, lowSat, lowVal, highHue, highSat, highVal], [GBx, GBy])
        
    if rsignDebug:
        RsDilateX = cv2.getTrackbarPos('RsDilateX', 'RsTest')
        if RsDilateX % 2 == 0:
            RsDilateX -= 1
        RsDilateY = cv2.getTrackbarPos('RsDilateY', 'RsTest')
        if RsDilateY % 2 == 0:
            RsDilateY -= 1
        MaxRsS = cv2.getTrackbarPos('MaxRsS', 'RsTest')
        MinRsS = cv2.getTrackbarPos('MinRsS', 'RsTest')
        MaxRsRatio = cv2.getTrackbarPos('MaxRsRatio', 'RsTest')/50
        em.SetRsignPara([RsDilateX, RsDilateY], MaxRsS, MaxRsRatio, MinRsS)

    if flowbarDebug:
        Rin = cv2.getTrackbarPos('Rin', 'FBTest')
        Rout = cv2.getTrackbarPos('Rout', 'FBTest')
        MaxlongSide = cv2.getTrackbarPos('MaxlongSide', 'FBTest')/10
        MinlongSide = cv2.getTrackbarPos('MinlongSide', 'FBTest')/10
        fbMinEDdis = cv2.getTrackbarPos('fbMinEDdis', 'FBTest')
        em.SetFlowBarPara(Rin, Rout, MaxlongSide, MinlongSide, fbMinEDdis)

    if armorDebug:
        rinRatio = cv2.getTrackbarPos('rinRatio', 'ArTest')/20
        sideRatio = cv2.getTrackbarPos('sideRatio', 'ArTest')/100
        em.SetArmorPara(rinRatio, sideRatio)
   
    if smallfanDebug:
        MaxdeltaAngle = cv2.getTrackbarPos('MaxdeltaAngle', 'SfanTest')
        MindeltaAngle = cv2.getTrackbarPos('MindeltaAngle', 'SfanTest')
        em.SetSmallFanPredictPara(MaxdeltaAngle, MindeltaAngle)

def DebugInit(preDebug, rsignDebug, flowbarDebug, armorDebug, smallfanDebug, pretreat, rsign, flowbar, armor, sfan):
    if preDebug:
        CreatePretreatmentTrackerBar(pretreat[0], pretreat[1])
    if rsignDebug:
        CreateRsignTrackerBar(rsign[0], rsign[1], rsign[2], rsign[3])
    if flowbarDebug:
        CreateFlowBarTrackerBar(flowbar[0], flowbar[1], flowbar[2], flowbar[3], flowbar[4])
    if armorDebug:
        CreateArmorTrackerBar(armor[0], armor[1])
    if smallfanDebug:
        CreateSmallfanTrackbar(sfan[0], sfan[1])
    