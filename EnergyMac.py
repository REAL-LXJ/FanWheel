'''
Author: LXJ
Date: 2021-03-29 11:08:28
LastEditTime: 2021-06-27 23:36:11
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \python_opencv\camera_test\EnergyMachine.py
'''
import cv2
import copy
import math
import numpy as np

class EnergyMac:
    
    def __init__(self,
                width,
                height,
                hsvPara,
                GBSize,
                RsDilatePara,
                MaxRsS,
                MaxRsRatio,
                MinRsS,
                Rin,
                Rout,
                MaxlongSide,
                MinlongSide,
                fbMinEDdis,
                rinRatio,
                sideRatio,
                MaxdeltaAngle,
                MindeltaAngle,
                hsvParaRed
                ):
        self.width = width
        self.height = height
        self.hsvPara = hsvPara
        self.GBSize = GBSize       
        self.RsDilatePara = RsDilatePara
        self.MaxRsS = MaxRsS
        self.MaxRsRatio = MaxRsRatio
        self.MinRsS = MinRsS
        self.Rin = Rin
        self.Rout = Rout
        self.MaxlongSide = MaxlongSide
        self.MinlongSide = MinlongSide
        self.fbMinEDdis = fbMinEDdis
        self.rinRatio = rinRatio
        self.sideRatio = sideRatio
        self.MaxdeltaAngle = MaxdeltaAngle
        self.MindeltaAngle = MindeltaAngle
        self.hsvParaRed = hsvParaRed
        self.hitAngle = 0
        self.lastAngle = 0
        self.lastpredictAngle = 0
        self.realAgHisList = np.array([0,0,0,0,0,0,0],'float64')
    
    def SwitchDebug(self, preDebug, rsignDebug, flowbarDebug, armorDebug, smallfanDebug):
        self.preDebug = preDebug
        self.rsignDebug = rsignDebug
        self.flowbarDebug = flowbarDebug
        self.armorDebug = armorDebug
        self.smallfanDebug = smallfanDebug
    
    def SetPretreatmentPara(self, hsvPara, GBSize):
        self.hsvPara = hsvPara
        self.GBSize = GBSize
    
    def SetRsignPara(self, RsDilatePara, MaxRsS, MaxRsRatio, MinRsS):
        self.RsDilatePara = RsDilatePara
        self.MaxRsS = MaxRsS
        self.MaxRsRatio = MaxRsRatio
        self.MinRsS = MinRsS

    def SetFlowBarPara(self, Rin, Rout, MaxlongSide, MinlongSide, fbMinEDdis):
        self.Rin = Rin
        self.Rout = Rout
        self.MaxlongSide = MaxlongSide
        self.MinlongSide = MinlongSide
        self.fbMinEDdis = fbMinEDdis

    def SetArmorPara(self, rinRatio, sideRatio):
        self.rinRatio = rinRatio
        self.sideRatio = sideRatio

    def SetSmallFanPredictPara(self, MaxdeltaAngle, MindeltaAngle):
        self.MaxdeltaAngle = MaxdeltaAngle
        self.MindeltaAngle = MindeltaAngle

    '''
    @description: get the hit point 
    @param {*} self
    @param {*} frame
    @return {*}
    '''    
    def GetHitPoint(self, frame, direct, size):
        
        self.__frame = frame
        self.__size = size#0小符， 1大符
        x, y, z = -1, -1, -1

        #! image pretreatment (HSV or RGB)
        mask = self.__Pretreatment_Process(frame)
        
        #! find R sign scope
        RsignScope = self.__FindRsignScope(mask)
        center = RsignScope[0]#R所在矩形框中点
        scope = RsignScope[1]#R所在矩形框的四个顶点
        #print(center)

        #! correct R position
        #! make the center X,Y coordinate perform some pixel offset
        Center = [center[0]+3, center[1]+7]
        
        #! find Flow Bar and Revised R sign Point
        correctCenter, flowBarList = self.__FindFlowBar(mask, Center, scope)
        #print(flowBarList)
        
        #! find distinguish Position
        armorList = self.__FindOriPos(mask, correctCenter, flowBarList)
        #print(armorList)
         
        #计算全向角，直接砍成线性去预测
        #1.取到未击打和转向下一块   使预测的半径R由r1->R->r2  (修正非平面度)
        #2.若取不到，则r1裸圆
        #3.帧差计算角速度给出预测
        #4.嵌入式给补偿去追

        #1.判断击打点象限等级
        #2.对于不同等级给不同预测角度
        
        #1.采集数据判断能量机关旋转1°-2°的速度
        #2.对于小符，速度是固定的 0.4-20°
        #3.对于大符，速度是周期变化，根据图像得出波峰波谷
        #4.预测角度跟速度挂钩

        #quadrant_level = self.__determine_quadrant_level(correctCenter, armorList)
        #print(quadrant_level)
        #direct = 1 #1顺时针 1逆时针
        #size = 1 #0小符 1大符
        #normal EM
        x, y, z = self.__NormalHit(correctCenter, armorList, direct)
    
        return x, y, z

    def __save_data(self, content, filename, mode = 'a+'):#存取列表数据
        with open(filename, mode, encoding='utf-8') as f:
            for i in range(len(content)):
                f.write(str(content[i])+ '\n')
            f.close()

    def __NormalHit(self, Center, armorList, direct):
        #x轴0 顺时针->360
        '''
        originalAngle = 20
        if quadrant_level == 1:#第一象限
            predictAngle = originalAngle - 2
        elif quadrant_level == 2:#第二象限
            predictAngle = originalAngle + 2
        elif quadrant_level == 3:#第三象限
            predictAngle = originalAngle + 6
        elif quadrant_level == 4:#第四象限
            predictAngle = originalAngle + 4
        elif quadrant_level == 5:
            predictAngle = originalAngle
        elif quadrant_level == 6:
            predictAngle = originalAngle
        elif quadrant_level == 7:
            predictAngle = originalAngle
        elif quadrant_level == 8:
            predictAngle = originalAngle
        else:
            predictAngle = originalAngle
        '''
        predictAngle = 20
        hitAngle = -1
        hitX = -1
        hitY = -1
        hitDis = -1
        angleDisList = []
        hitAngleList = []
        for i in range(len(armorList)):
            angle = 0
            flag = armorList[i][-1]#是否击打标志，1待击打，0已击打
            x, y = armorList[i][0], armorList[i][1]#装甲板中心坐标
            vectorX = x - Center[0]
            vectorY = y - Center[1]
            if vectorX > 0 and vectorY > 0:
                angle = math.atan(abs(vectorY/vectorX))*180/math.pi
            elif vectorX < 0 and vectorY > 0:
                angle = 180 - math.atan(abs(vectorY/vectorX))*180/math.pi
            elif vectorX < 0 and vectorY < 0:
                angle = 180 + math.atan(abs(vectorY/vectorX))*180/math.pi
            elif vectorX > 0 and vectorY < 0:
                angle = 360 - math.atan(abs(vectorY/vectorX))*180/math.pi
            elif vectorX == 0:
                if vectorY > 0:
                    angle = 270
                else:
                    angle = 90
            elif vectorY == 0:
                if vectorX > 0:
                    angle = 0
                else:
                    angle = 180
            dis = pow(vectorX**2+vectorY**2,0.5)#向量长
            if flag == 0:#已击打扇叶
                angleDisList.append([dis,angle])
                #print(angleDisList)
            if flag == 1:#待击打扇叶
                hitAngle = angle
                hitAngleList.append(hitAngle)
                #self.__save_data(hitAngleList, 'smallfanhitAngle.txt')#存取小符待击打角
                #self.__save_data(hitAngleList, 'bigfanhitAngle.txt')#存取大符待击打角
                #print(hitAngleList)
                hitX, hitY = x, y
            cv2.putText(self.__frame, str(int(angle)), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)#全向角
        if hitX == -1:
            return -1,-1,-1
        hitDis = self.EuclideanDistance([hitX, hitY], Center)
        correctFlag = False                                      
        
        #print("-______________________-")
        hhitAngle = hitAngle

        if self.__size == 1:
            predictAngle = self.__findRealTarAngle(hitAngle)
            self.lastAngle = hitAngle 

        for j in range(len(angleDisList)):
            #print(angleDisList[j][1],hitAngle)
            deltaAngle = angleDisList[j][1] - hhitAngle
            #print(hitAngle)
            if deltaAngle > self.MindeltaAngle and deltaAngle < self.MaxdeltaAngle:#! 用角偏差范围去修正预测点，72°上下调 
                hitDis = self.EuclideanDistance([hitX, hitY], Center)+direct*predictAngle*(self.EuclideanDistance([hitX, hitY], Center)-angleDisList[j][0])/72
                #print(hitAngle)
                hitAngle -= predictAngle*direct
                #print(hitAngle)
                if hitAngle < 0:
                    hitAngle += 360
                if hitAngle > 360:
                    hitAngle -= 360
                correctFlag = True

            if deltaAngle+360 > self.MindeltaAngle and deltaAngle+360 < self.MaxdeltaAngle:    
                hitDis = self.EuclideanDistance([hitX, hitY], Center)+direct*predictAngle*(self.EuclideanDistance([hitX, hitY], Center)-angleDisList[j][0])/72
                hitAngle -= predictAngle*direct
                if hitAngle < 0:
                    hitAngle -= 360
                if hitAngle > 360:
                    hitAngle -= 360
                correctFlag = True
                
        #print(correctFlag)
        if not correctFlag:
            hitAngle -= predictAngle*direct
            if hitAngle < 0:
                hitAngle += 360
            if hitAngle > 360:
                hitAngle -= 360
        if hitAngle > 270:
            #math.tan = x/y  x^2+y^2 = hitDis
            #x = y*math.tan  y = 根号hitDis/根号(1+math.tan^2)
            vx = hitDis/pow(1+math.tan(hitAngle*math.pi/180)**2,0.5) 
            vy = vx*math.tan(hitAngle*math.pi/180)
            if vy > 0:
                vy = -vy
            if vx < 0:
                vx = -vx
        elif hitAngle > 180:
            vx = hitDis/pow(1+math.tan(hitAngle*math.pi/180)**2,0.5) 
            vy = vx*math.tan(hitAngle*math.pi/180)
            if vy > 0:
                vy = -vy
            if vx > 0:
                vx = -vx
        elif hitAngle > 90:
            vx = hitDis/pow(1+math.tan(hitAngle*math.pi/180)**2,0.5) 
            vy = vx*math.tan(hitAngle*math.pi/180)
            if vy < 0:
                vy = -vy
            if vx > 0:
                vx = -vx
        else:
            vx = hitDis/pow(1+math.tan(hitAngle*math.pi/180)**2,0.5) 
            vy = vx*math.tan(hitAngle*math.pi/180)
            if vy < 0:
                vy = -vy
            if vx < 0:
                vx = -vx
        hitX = vx + Center[0]
        hitY = vy + Center[1]
        cv2.circle(self.__frame, (int(hitX),int(hitY)),5,(255,255,255),-1)
        if self.smallfanDebug:
            #cv2.circle(self.__frame, (int(Center[0]),int(Center[1])), 11, (255,255,255),-1)
            cv2.imshow("SfanTest",self.__frame)
        #print(hitAngle)
        return hitX, hitY, hitDis
        
    def __findRealTarAngle(self, angle):#y=0.785sin(1.884t)+1.305
        deltaAg = self.lastAngle - angle
        '''
        if deltaAg < 0:
            direct = -1
        elif deltaAg > 0:
            direct = 1
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        #print(deltaAg)
        #ΔAngle<0,顺;ΔAngle>0,逆 
        if deltaAg > 350:
            deltaAg -= 360
        elif deltaAg < -350:
            deltaAg += 360
        if abs(deltaAg)<1.3:#show_image里的角度为2
            self.hitAngle = deltaAg*0.1 + 0.9*self.hitAngle#滤波
        self.realAgHisList[0:6] = self.realAgHisList[1:7]
        self.realAgHisList[6] = self.hitAngle
        #采集数据画图观察
        peak = 0.8#波峰
        trough = 0.3#波谷
        sfanPreAngle = 20#小符预测角度
        DPF = 0.4#degree per frame每帧角度
        k = np.sum(self.realAgHisList[0:2])/(np.sum(self.realAgHisList[5:7])+0.0001)
        v = abs(np.average(self.realAgHisList))
        #cv2.putText(self.__frame, "k:"+str(round(k,2)), (250,320),font,1.2,(255,255,255),2)
        #cv2.putText(self.__frame, "v:"+str(round(v,2)), (250,360),font,1.2,(255,255,255),2)

        '''
        #1.LSF:斜率比较小，转的非常快; 2.LSS：斜率比较小，转的非常慢； 3.HSTF：斜率比较大，转速变快； 4. HSTS：斜率比较大，转速变慢
        HSTF_rate = peak - 0.2# high_slope_turn_fast 斜率比较大，转速变快
        LSS_rate = peak - 0.6# low_slope_slow 斜率比较小，转的非常慢
        HSTS_rate = trough + 0.2# high_slope_turn_slow 斜率比较大，转速变慢
        LSF_rate = trough # low_slope_fast 斜率比较小，转的非常快 

        HSTF_predictAngle = HSTF_rate * (sfanPreAngle/DPF)#  0.6*(20/0.4) = 30    
        LSS_predictAngle = LSS_rate * (sfanPreAngle/DPF)#  0.2*(20/0.4) = 10
        HSTS_predictAngle = HSTS_rate * (sfanPreAngle/DPF)#  0.5*(20/0.4) = 25
        LSF_predictAngle = LSF_rate * (sfanPreAngle/DPF)#  0.3*(20/0.4) = 15

        mid_extent = 0.5*(peak+trough)# 0.5*(0.8+0.3) = 0.55
        HSTF_to_LSS_extent = mid_extent #0.55
        LSS_to_HSTS_extent = mid_extent + 0.25 #0.8
        HSTS_to_LSF_extent = mid_extent - 0.25 #0.3
        '''

        if v < 0.35:
            cv2.putText(self.__frame, "slow", (250,400),font,1.2,(255,255,255),2)
            predictAg = 10
        elif v > 1.0:
            cv2.putText(self.__frame, "fast", (250,400),font,1.2,(255,255,255),2)
            predictAg = 25
        elif k > 1.05 and k < 1.4 and v > 0.65:
            cv2.putText(self.__frame, "being slow", (250,400),font,1.2,(255,255,255),2)
            predictAg = 10
        elif k < 0.95 and k > 0.75 and v < 0.45:
            cv2.putText(self.__frame, "being fast", (250,400),font,1.2,(255,255,255),2)
            predictAg = 25
        else:
            predictAg = self.lastpredictAngle
        self.lastpredictAngle = predictAg
        #print(self.hitAngle)
        
        '''
        PeakrRate = peak - 0.2#比peak小
        TroughRate = trough + 0.1#比trough大
        extent = 0.5#(peak+trough)/2
        predictPeak = PeakrRate*(sfanPreAngle/DPF)#24,也可为确定的度数
        predictTrough = TroughRate*(sfanPreAngle/DPF)#9 
        '''
        '''
        if abs(self.hitAngle) <= HSTS_to_LSF_extent:
            predictAg = LSF_predictAngle
        elif abs(self.hitAngle) > HSTS_to_LSF_extent and abs(self.hitAngle) <= HSTF_to_LSS_extent:
            predictAg = HSTS_predictAngle
        elif abs(self.hitAngle) > HSTF_to_LSS_extent and abs(self.hitAngle) <= LSS_to_HSTS_extent:
            predictAg = HSTF_predictAngle
        elif abs(self.hitAngle) > LSS_to_HSTS_extent:
            predictAg = LSS_predictAngle
        '''

        '''
        if abs(self.hitAngle) <= extent:
            predictAg = predictTrough
        else:
            predictAg = predictPeak
        #print(predictAg)
        '''

        
        return predictAg
    '''
    def __predict_hit_point(self, correctCenter, armorList):
        for i in range(len(armorList)):
            angle = 20
            hit_flag = armorList[i][-1]
            armor_center_x, armor_center_y = armorList[i][0], armorList[i][1]
            R_center_x = correctCenter[0]
            R_center_y = correctCenter[1]
            if hit_flag == 1:
                xside = -float(armor_center_x - R_center_x)
                yside = -float(armor_center_y - R_center_y)
                PredictPointX = R_center_x + math.cos(angle) * xside - math.sin(angle) * yside
                PredictPointY = R_center_y + math.sin(angle) * xside + math.cos(angle) * yside
                cv2.circle(self.__frame, (int(PredictPointX), int(PredictPointY)), 10, (0, 127, 255), -1)

            return PredictPointX, PredictPointY
    '''

    '''
    @description: make the em is divided into 8 regions and configured with different levels under the condition of 4 quadrants
    @param {*} self
    @param {*} correctCenter
    @param {*} armorList
    @return {*}
    '''    
    def __determine_quadrant_level(self, correctCenter, armorList):
        quadrant_level = -1
        for i in range(len(armorList)):
            hit_flag = armorList[i][-1]
            armor_center_x, armor_center_y = armorList[i][0], armorList[0][1]
            R_center_x = correctCenter[0]
            R_center_y = correctCenter[1]
            if hit_flag == 1:
                if((armor_center_x > R_center_x) and (armor_center_y < R_center_y)):#第一象限
                    quadrant_level = 1
                elif((armor_center_x > R_center_x) and (armor_center_y > R_center_y)):#第二象限
                    quadrant_level = 2
                elif((armor_center_x < R_center_x) and (armor_center_y > R_center_y)):#第三象限
                    quadrant_level = 3
                elif((armor_center_x < R_center_x) and (armor_center_y < R_center_y)):#第四象限
                    quadrant_level = 4
                elif((armor_center_x == R_center_x) and (armor_center_y < R_center_y)):#y正半轴
                    quadrant_level = 5
                elif((armor_center_x > R_center_x) and (armor_center_y == R_center_y)):#x正半轴
                    quadrant_level = 6
                elif((armor_center_x == R_center_x) and (armor_center_y > R_center_y)):#y负半轴
                    quadrant_level = 7
                elif((armor_center_x < R_center_x) and (armor_center_y == R_center_y)):#x负半轴
                    quadrant_level = 8
                else:
                    quadrant_level = -1
            
        return quadrant_level

    def __FindOriPos(self, frame, Center, flowBarList):
        armorList = []
        Rthird = self.Rin * self.rinRatio
        mask = np.zeros_like(frame)
        cv2.circle(mask, (int(Center[0]),int(Center[1])), int(Rthird), (255,255,255),30)
        #cv2.imshow("thirdmask", mask)
        maskFinal = cv2.bitwise_and(mask, frame)#装甲板外边轮廓
        if self.armorDebug:
            cv2.imshow("Armormask", maskFinal)
        #cv2.circle(self.__frame, (int(Center[0]),int(Center[1])), int(self.Rin*3.5), (255,255,255),30)
        #cv2.imshow('m',self.__frame)
        contours, hierarchy = cv2.findContours(maskFinal,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #print("_______________________________________")
        contoursLength = len(contours)
        for c in range(contoursLength):
            center, size, angle = cv2.minAreaRect(contours[c])
            #cv2.circle(self.__frame, (int(center[0]), int(center[1])), 4, (0, 0, 127), -1)
            #cv2.circle(self.__frame, (int(Center[0]), int(Center[1])), 4, (0, 0, 127), -1)
            #vertices = cv2.boxPoints((center, size, angle))
            #vertices = np.int0(vertices)
            #cv2.drawContours(self.__frame, [vertices], 0, (255, 0, 255), 2)

            shortSide = min(size[0],size[1])/2
            if center[0] == Center[0]:
                x = 0
                y = shortSide
            else:
                deltaY = (center[1]-Center[1])
                deltaX = (center[0]-Center[0])
                k = (deltaY+0.0001)/(deltaX+0.0001) #斜率
                #shortSide^2 = Δx^2 + Δy^2
                #Δy = k*Δx
                #四象限
               
                if deltaX >= 0 and deltaY >= 0:                  #第一象限
                    x = pow(shortSide*shortSide/(1+k*k),0.5)
                    y = pow(shortSide*shortSide/(1+1/(k*k)),0.5)
                if deltaX>=0 and deltaY<0:                       #第二象限
                    x = pow(shortSide*shortSide/(1+k*k),0.5)
                    y = -pow(shortSide*shortSide/(1+1/(k*k)),0.5)
                if deltaX<0 and deltaY<0:                        #第三象限
                    x = -pow(shortSide*shortSide/(1+k*k),0.5)
                    y = -pow(shortSide*shortSide/(1+1/(k*k)),0.5)
                if deltaX<0 and deltaY>=0:                       #第四象限
                    x = -pow(shortSide*shortSide/(1+k*k),0.5)
                    y = pow(shortSide*shortSide/(1+1/(k*k)),0.5)
            K = self.sideRatio
            x0 = center[0]+x
            y0 = center[1]+y
            x1 = Center[0]+K*(x0-Center[0])#装甲板中心X坐标
            y1 = Center[1]+K*(y0-Center[1])#装甲板中心Y坐标
            dis = 9999
            situa = -1

            for b in flowBarList:
                #cv2.line(self.__frame, (int(b[0]), int(b[1])), (int(x1), int(y1)), (0, 255, 0), 2)#连接流动条和装甲板所在点
                #print(self.EuclideanDistance([b[0],b[1]],[x1,y1]))
                if(self.EuclideanDistance([b[0],b[1]],[x1,y1]) < dis):#利用流动条和装甲板所在点的欧式距离去匹配流动条扇叶装甲板 #! 欧式距离 
                    dis = self.EuclideanDistance([b[0],b[1]],[x1,y1])
                    situa = b[-1]
            if situa == 1:#待击打扇叶装甲板(流动条扇叶装甲板)    
                cv2.circle(self.__frame, (int(x1),int(y1)),5,(0,255,0),-1)
                armorList.append([x1,y1,1])
                #cv2.putText(self.__frame,"dis: "+str(dis*10),(100,200),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)

            if situa == 0:#已击打扇叶装甲板(非流动条扇叶装甲板)
                cv2.circle(self.__frame, (int(x1),int(y1)),5,(0,0,255),-1)
                armorList.append([x1,y1,0])
            #cv2.circle(self.__frame, (int(center[0]+x),int(center[1]+y)),5,(0,0,255),-1)
            #cv2.circle(self.__frame, (int(center[0]),int(center[1])),5,(0,255,255),-1)
            #cv2.line(self.__frame, tuple(vertices[0]), tuple(vertices[1]), (0,0,255), 2)
            #cv2.line(self.__frame, tuple(vertices[1]), tuple(vertices[2]), (0,0,255), 2)
            #cv2.line(self.__frame, tuple(vertices[2]), tuple(vertices[3]), (0,0,255), 2)
            #cv2.line(self.__frame, tuple(vertices[3]), tuple(vertices[0]), (0,0,255), 2)

        if self.armorDebug: 
            cv2.imshow('ArTest',self.__frame)
        return armorList
    
    def __FindFlowBar(self, frame, Center, scope):
        #蒙版圆！！！
        flowBarList = []
        correctCenter = Center
        Rin = self.Rin #! 内蒙版圆半径
        Rout = self.Rout #! 外蒙版圆半径
        maskIn = copy.copy(frame)
        maskOut = copy.copy(frame)
        black = (0,0,0)
        #white = (255,255,255)
        if Center[0] != -1:#已确定R范围
            maskIn = cv2.circle(maskIn, (int(Center[0]), int(Center[1])), Rin, black, -1)
            maskOut = cv2.circle(maskOut, (int(Center[0]), int(Center[1])), Rout, black, -1)
            maskFinal = cv2.bitwise_xor(maskIn, maskOut)#获取部分流动条所在圆环(三道灯条，中间为流动条，两侧为非流动条)
            if self.flowbarDebug:
                cv2.imshow('FlowBarmask', maskFinal)
            contours, hierarchy = cv2.findContours(maskFinal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            realFlow = []#流动条列表(中间灯条)
            notFlow = []#非流动条列表(流动条两侧灯条)
            drawV = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
            
            contoursLength = len(contours)
            for c in range(contoursLength):
                center, size, angle = cv2.minAreaRect(contours[c])
                vertices = cv2.boxPoints((center, size, angle))
                #vertices = np.int0(vertices)
                #cv2.drawContours(self.__frame, [vertices], 0, (255, 0, 255), 2)
                #得到长边和中心点垂直短边向量
                if size[0]<size[1]:
                    vector = center-(vertices[0]+vertices[3])/2
                    longSide = size[1]
                    shortSide = size[0]
                else:
                    vector = center-(vertices[0]+vertices[1])/2
                    longSide = size[0]
                    shortSide = size[1]
                #计算向量和水平轴夹角
                #angleHori = int(math.atan2(vector[1], vector[0]) * 180/math.pi)

                #! 根据长边最大值和长边最小值筛选出流动条所在区域(圆环范围中间为流动条所在区域，两侧为灯条其他区域)
                if longSide < self.MaxlongSide and longSide > self.MinlongSide:
                    drawVector = -vector + center
                    drawV = vertices
                    drawL = int(longSide)
                    x0, y0 = center[:]
                    x1, y1 = drawVector[:]
                    realFlow.append([x0,y0,x1,y1])
                else:
                    notFlow.append(center)
            
            
            font = cv2.FONT_HERSHEY_SIMPLEX 
            if self.flowbarDebug:
                if drawV[0][0] != -1:
                    #cv2.line(self.__frame, tuple(drawV[0]), tuple(drawV[1]), (255,0,0), 2)
                    #cv2.line(self.__frame, tuple(drawV[1]), tuple(drawV[2]), (255,0,0), 2)
                    #cv2.line(self.__frame, tuple(drawV[2]), tuple(drawV[3]), (255,0,0), 2)
                    #cv2.line(self.__frame, tuple(drawV[3]), tuple(drawV[0]), (255,0,0), 2)
                    cv2.putText(self.__frame, "L:"+str(drawL), (50,100),font,1.2,(255,255,255),2)
                else:
                    cv2.putText(self.__frame, "False", (50,100),font,1.2,(255,255,255),2)
            


                
            if len(realFlow) == 1:#屏幕中存在一个流动条，BGR
                flowBarList.append([realFlow[0][0],realFlow[0][1],1])#未击打流动条标志为1
                cv2.circle(self.__frame,(int(realFlow[0][0]),int(realFlow[0][1])),8,(0,255,0),-1)#在未击打流动条上画绿点
                cv2.putText(self.__frame, "0", (int(realFlow[0][0]),int(realFlow[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)#在未击打流动条上标0
            if len(realFlow) > 1: 
                for i in range(len(realFlow)):
                    num = 0
                    for j in range(len(notFlow)):#遍历所有的非流动条
                        cv2.circle(self.__frame,(int(notFlow[j][0]),int(notFlow[j][1])),8,(0,255,255),-1)#在非流动条上画黄点
                        #cv2.line(self.__frame, (int(realFlow[i][0]), int(realFlow[i][1])), (int(notFlow[j][0]), int(notFlow[j][1])), (0, 255, 0), 2)#连接流动条和非流动条所在点
                        #print(self.EuclideanDistance([realFlow[i][0],realFlow[i][1]],[notFlow[j][0],notFlow[j][1]]))
                        if self.EuclideanDistance([realFlow[i][0],realFlow[i][1]],[notFlow[j][0],notFlow[j][1]]) < self.fbMinEDdis: #利用流动条和非流动条所在点的欧式距离区分已击打流动条和未击打流动条#! 欧式距离
                            num += 1
                    if num == 2:
                        flowBarList.append([realFlow[i][0],realFlow[i][1],0])#已击打流动条标志为0
                        cv2.circle(self.__frame,(int(realFlow[i][0]),int(realFlow[i][1])),8,(0,0,255),-1)#已击打流动条画红点
                        cv2.putText(self.__frame, str(num), (int(realFlow[i][0]),int(realFlow[i][1])),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
                    else:
                        flowBarList.append([realFlow[i][0],realFlow[i][1],1])#未击打流动条标志为1
                        cv2.circle(self.__frame,(int(realFlow[i][0]),int(realFlow[i][1])),8,(0,255,0),-1)#在未击打流动条上画绿点
                        cv2.putText(self.__frame, str(num), (int(realFlow[i][0]),int(realFlow[i][1])),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
                        realFlow[i] = [0,0,0,0]
                    num = 0
                correctX, correctY = self.Findintersection(realFlow, scope)
                if correctX != -1:
                    correctCenter = [correctX, correctY]
                #print("Corr",correctX,correctY)
                #print("Ori",Center[0],Center[1])
                #cv2.circle(self.__frame,(int(correctX),int(correctY)),1,(0,0,255),-1)
 
        if self.flowbarDebug:  
            cv2.imshow('FBTest', self.__frame)
            #cv2.imshow('maskIn', maskIn)
            #cv2.imshow('maskOut', maskOut)
            #cv2.imshow('final',maskFinal)
        
        return correctCenter, flowBarList
    
    def EuclideanDistance(self,c,c0):
        '''
        计算欧氏距离
        @para c(list):[x, y]
        @para c0(list):[x, y]
        @return double:欧氏距离
        '''
        return pow((c[0]-c0[0])**2+(c[1]-c0[1])**2, 0.5)
    
    def Findintersection(self, pointList, scope):
        '''
        y=(y2-y1)*(x-x1)/(x2-x1)+y1
        y=(Y2-Y1)*(x-X1)/(X2-X1)+Y1
        (y2-y1)*(x-x1)/(x2-x1)+y1 =
        (Y2-Y1)*(x-X1)/(X2-X1)+Y1
        ((y2-y1)/(x2-x1))*x - ((y2-y1)/(x2-x1))*x1 + y1=
        ((Y2-Y1)/(X2-X1))*x - ((Y2-Y1)/(X2-X1))*X1 + Y1
        x = (- ((Y2-Y1)/(X2-X1))*X1 + Y1 -y1 +((y2-y1)/(x2-x1))*x1)/(((y2-y1)/(x2-x1))-((Y2-Y1)/(X2-X1)))
        '''
        centerListX = []
        centerListY = []
        left, right, top, bot = scope[:]
        for i in range(len(pointList)):
            x1, y1, x2, y2 = pointList[i][:]
            if x1 != x2:
                for j in range(i+1,len(pointList)):
                    X1, Y1, X2, Y2 = pointList[j][:]
                    if X1 != X2:
                        x = (- ((Y2-Y1)/(X2-X1))*X1 + Y1 -y1 +((y2-y1)/(x2-x1))*x1)/(((y2-y1)/(x2-x1))-((Y2-Y1)/(X2-X1)))
                        y = (y2-y1)*(x-x1)/(x2-x1)+y1
                        if x > left and x < right and y > top and y < bot:
                            centerListX.append(x)
                            centerListY.append(y)
        if len(centerListX) == 0:
            return -1, -1
        return np.average(np.array(centerListX)), np.average(np.array(centerListY))
    
    def __FindRsignScope(self, frame):
        #print(self.RsBlurPara)
        #Blur = cv2.medianBlur(frame, self.RsBlurPara)
        frame = cv2.dilate(frame, (self.RsDilatePara[0], self.RsDilatePara[1]))
        
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        #[center, leftPointX, rightPointX, topPointY, bottomPointY]
        RsignList = [[-1, -1], [-1, -1, -1, -1]]
        drawV = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
        drawS = -1
        drawRatio = -1
        contoursLength = len(contours)
        #print('-----------------------------------')
        for c in range(contoursLength):
            
            contoursFine = True
            
            center, size, angle = cv2.minAreaRect(contours[c])
            vertices = cv2.boxPoints((center, size, angle))
            #得到长边和中心点垂直短边向量
            if size[0]<size[1]:
                #vector = center-(vertices[0]+vertices[3])/2
                longSide = size[1]
                shortSide = size[0]
            else:
                #vector = center-(vertices[0]+vertices[1])/2
                longSide = size[0]
                shortSide = size[1]
            
            if longSide*shortSide > self.MaxRsS:
                contoursFine = False
            if longSide*shortSide < self.MinRsS:
                contoursFine = False
            if longSide/(shortSide+0.0001) > self.MaxRsRatio:
                contoursFine = False
            
            #print("S: ", longSide*shortSide)
            #print("Ratio: ", longSide/shortSide)
            if contoursFine:
                leftPointX = np.min(vertices[:, 0])
                rightPointX = np.max(vertices[:, 0])
                topPointY = np.min(vertices[:, 1])
                bottomPointY = np.max(vertices[:, 1])
                RsignList = [center, [leftPointX, rightPointX, topPointY, bottomPointY]]
                drawV = vertices
                drawS = int(longSide*shortSide)#面积
                drawRatio = int(longSide/shortSide+0.0001)#长宽比
                break
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.rsignDebug:
            if drawV[0][0] != -1:
                #cv2.line(self.__frame, tuple(drawV[0]), tuple(drawV[1]), (0,0,255), 2)
                #cv2.line(self.__frame, tuple(drawV[1]), tuple(drawV[2]), (0,0,255), 2)
                #cv2.line(self.__frame, tuple(drawV[2]), tuple(drawV[3]), (0,0,255), 2)
                #cv2.line(self.__frame, tuple(drawV[3]), tuple(drawV[0]), (0,0,255), 2)
                cv2.putText(self.__frame, "S:"+str(drawS), (50,200),font,1.2,(255,255,255),2)
                cv2.putText(self.__frame, "Ratio:"+str(drawRatio), (50,300),font,1.2,(255,255,255),2)
            else:
                cv2.putText(self.__frame, "False", (50,200),font,1.2,(255,255,255),2)
            cv2.imshow('RsTest', self.__frame)
            cv2.imshow('RsTestBi', frame)
            
        return RsignList
                
    def __Pretreatment_Process(self, frame):
        '''         返回二值蒙版
        @para cv.mat: 原始帧bgr图像
        @return cv.mat: 白色为感兴趣的色彩
        '''     
        #cv2.imshow("frame", frame)    
        frame = cv2.GaussianBlur(frame,(self.GBSize[0], self.GBSize[1]),0) 
        '''
        channel = cv2.split(frame)
        red = cv2.subtract(channel[2], channel[0])
        #blue = cv2.subtract(channel[0],channel[2])
        _, threshold = cv2.threshold(red, 75, 255, cv2.THRESH_BINARY)   
        cv2.imshow("RGB", threshold)  
        '''
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.hsvPara[0:3]), np.array(self.hsvPara[3:6]))
        #cv2.imshow("HSV", mask)
        #mask = threshold
        
        if self.preDebug:
            cv2.imshow("colorTest", mask)
        return mask
        






