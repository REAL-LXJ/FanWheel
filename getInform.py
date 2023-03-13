'''
Author: LXJ
Date: 2021-03-29 08:28:27
LastEditTime: 2021-03-29 08:28:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \python_opencv\camera_test\getInform.py
'''
import json

'''
@description: 
@param {*} jsonFilePath
@return {*}
'''
def getInform(jsonFilePath):
    print("Load information from settings.json ...")
    with open(jsonFilePath, 'r', encoding= 'utf-8')as load_f:

        load_dict = json.load(load_f, strict = False)

        #choose camera or video to operate program
        camera = load_dict["ModeSwitch"]["camera"]
        video = load_dict["ModeSwitch"]["video"]   
        
        videoPath = load_dict["Path"]["videoPath"]

        width = load_dict["CameraSet"]["width"]
        height = load_dict["CameraSet"]["height"]
        exposure = load_dict["CameraSet"]["exposure"]
        showOriC = load_dict["CameraSet"]["showOri"]

        speed = load_dict["VideoSet"]["speed"]
        showOriV = load_dict["VideoSet"]["showOri"]

        pretreatment = load_dict["DebugSwitch"]["pretreatment"] 
        rsign = load_dict["DebugSwitch"]["rsign"]
        flowbar = load_dict["DebugSwitch"]["flowbar"]
        armor = load_dict["DebugSwitch"]["armor"]
        smallfan = load_dict["DebugSwitch"]["smallfanpredict"]


        lowHblue = load_dict["hsvSet"]["hsvBlue"]["lowHue"]
        lowSblue = load_dict["hsvSet"]["hsvBlue"]["lowSat"]
        lowVblue = load_dict["hsvSet"]["hsvBlue"]["lowVal"]
        highHblue = load_dict["hsvSet"]["hsvBlue"]["highHue"]
        highSblue = load_dict["hsvSet"]["hsvBlue"]["highSat"]
        highVblue = load_dict["hsvSet"]["hsvBlue"]["highVal"]

        lowHred = load_dict["hsvSet"]["hsvRed"]["lowHue"]
        lowSred = load_dict["hsvSet"]["hsvRed"]["lowSat"]
        lowVred = load_dict["hsvSet"]["hsvRed"]["lowVal"]
        highHred = load_dict["hsvSet"]["hsvRed"]["highHue"]
        highSred = load_dict["hsvSet"]["hsvRed"]["highSat"]
        highVred = load_dict["hsvSet"]["hsvRed"]["highVal"]

        gblurx = load_dict["gblurSet"]["xsize"]
        gblury = load_dict["gblurSet"]["ysize"] 

        rsDilateX = load_dict["RsignSet"]["rsDilate"]["rsDilateX"]
        rsDilateY = load_dict["RsignSet"]["rsDilate"]["rsDilateY"]
        MaxRsS = load_dict["RsignSet"]["MaxRsS"]
        MinRsS = load_dict["RsignSet"]["MinRsS"]
        MaxRsRatio = load_dict["RsignSet"]["MaxRsRatio"]

        Rin = load_dict["FlowBarSet"]["Rin"]
        Rout = load_dict["FlowBarSet"]["Rout"]
        MaxlongSide = load_dict["FlowBarSet"]["MaxlongSide"]
        MinlongSide = load_dict["FlowBarSet"]["MinlongSide"]
        fbMinEDdis = load_dict["FlowBarSet"]["fbMinEDdis"]

        rinRatio = load_dict["ArmorSet"]["rinRatio"]
        sideRatio = load_dict["ArmorSet"]["sideRatio"]

        MaxdeltaAngle = load_dict["SmallFanPredictSet"]["MaxdeltaAngle"]
        MindeltaAngle = load_dict["SmallFanPredictSet"]["MindeltaAngle"]

        
    print("Load finished!")

    return camera, video, videoPath, \
        width, height, exposure, showOriC, \
        speed, showOriV, \
        pretreatment, rsign, flowbar, armor, smallfan, \
        lowHblue, lowSblue, lowVblue, highHblue, highSblue, highVblue, \
        lowHred, lowSred, lowVred, highHred, highSred, highVred, \
        gblurx, gblury, \
        rsDilateX, rsDilateY, MaxRsS, MinRsS, MaxRsRatio, \
        Rin, Rout, MaxlongSide, MinlongSide, fbMinEDdis, \
        rinRatio, sideRatio, \
        MaxdeltaAngle, MindeltaAngle
        