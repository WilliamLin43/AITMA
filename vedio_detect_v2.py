import numpy as np
import cv2

def croppoly(img,p):
    #global points
    pts=np.asarray(p)
    pts = pts.reshape((-1,1,2))
    ##Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped = img[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.fillPoly(mask,pts=[pts],color=(255,255,255))
    #cv2.copyTo(cropped,mask)
    dst=cv2.bitwise_and(cropped,cropped,mask=mask)
    bg=np.ones_like(cropped,np.uint8)*255 #fill the rest with white
    cv2.bitwise_not(bg,bg,mask=mask)
    dst2=bg+dst
    return dst2


def init_feature(cap,points): #初始化特徵點
    #設定特徵取樣參數
    feature_params = dict( maxCorners = 100, qualityLevel =0.3, minDistance =7, blockSize =7)
    lk_params = dict(winSize = (15,15), maxLevel = 5, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
    color = np.random.randint(0,255,(100,3)) #隨機色彩
    
    ret, old_frame = cap.read()  #讀取影片
    #old_frame = cv2.rotate(old_frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
    #cv2.flip(old_frame,1)
    #print(points)
    dst=croppoly(old_frame,points)
    #cv2.imshow("image", dst) #顯示遮罩後照片
    #cv2.waitKey(0)
    
    old_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) #轉灰階
    
    
    
    #x,y= old_gray.shape #取得frame大小
    
    
    #mask1 = np.zeros([x, y], dtype=np.uint8) #新增遮罩陣列
    #mask1[50:300, 600:y-50] = 255 #設定特徵取樣範圍
    #mask1[5:100, 5:200] = 255 #設定特徵取樣範圍
    
    #cv2.imshow("mask1", mask1) #顯示遮罩
    #cv2.waitKey(0)

    #old_gray1 = cv2.add(old_gray, np.zeros(np.shape(old_gray), dtype=np.uint8), mask=mask1)
    #cv2.imshow("image", old_gray) #顯示遮罩後照片
    #cv2.waitKey(0)
    
    p0 = cv2.goodFeaturesToTrack(old_gray, mask =None, **feature_params) #取特徵點
    #print(p0)
    #mask = np.zeros_like(old_frame)
    #cap.release()
    
    return feature_params,lk_params,color,old_gray,p0


def tracert_feature(cap,feature_params,lk_params,color,old_gray,p0,points):
    count = 0
   
    
    while(1):
        
        
        ret,frame = cap.read()
        #frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        dst=croppoly(frame,points)
        
        frame_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)        
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        mask = np.zeros_like(dst)


        if not st is None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        if st is None: #特徵點已移出畫面, 重新擷取特徵點
            feature_params,lk_params,color,old_gray,p0 = init_feature(cap,points)            
            tracert_feature(cap,feature_params,lk_params,color,old_gray,p0,points)
        
        '''    
        #設定更新特徵點
        if count > 120:
            #cap.release()
            print('Stop Move')
            count = 0
            move_flag = 0
            feature_params,lk_params,color,old_gray,p0,mask = init_feature(cap)
            #tracert_feature(cap,feature_params,lk_params,color,old_gray,p0,mask)
         '''  
            
            
        total_data_x = []
        total_data_y = []
        count_i = 1
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #print(str(i)+',a-c')
            #print(a-c)
            #print(str(i)+',b-d')
            #print(b-d)
            count_i +=1
            total_data_x.append((a-c))
            total_data_y.append((b-d))
            mask = cv2.line(mask, (a,b),(c,d),color[i].tolist(),2) #畫移動軌跡線
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1) #畫特徵點
        
        #print('X Value: '+str(total_data_x/count_i))
        #print('Y Value: '+str(total_data_y/count_i))
        if len(total_data_x) > 0:
            x_mean = sum(total_data_x)/len(total_data_x)
            y_mean = sum(total_data_y)/len(total_data_y)

        img = cv2.add(dst,mask) #顯示特徵移動軌跡
        

        
        if (x_mean) < 0.05 and x_mean > -0.05 and not st is None and (x_mean) != 0:
            print('Stop Move')
            cv2.putText(img,'Stop Move',(50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
            print('x value:'+str(x_mean))
            print('y value:'+str(y_mean))
        else:
            print('Move forward')
            cv2.putText(img,'Move forward',(50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
            print('x value:'+str(x_mean))

            if (y_mean) < -3.9  and (x_mean) < 0.05 and not st is None:
                print('Turn left')                
                cv2.putText(img,'Turn left',(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
                print('y value:'+str(y_mean))
    
            if (y_mean) > -2.5 and (x_mean) < 0.05 and not st is None:
                print('Turn right')                
                cv2.putText(img,'Turn right',(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
                print('y value:'+str(y_mean))
            
            
            
            
        cv2.putText(img,'x value:'+str((x_mean)) ,(50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
        cv2.putText(img,'y value:'+str((y_mean)) ,(50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)

            
        
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k ==27:
            break
        
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        count +=1


if __name__ == '__main__':
    
    cap = cv2.VideoCapture('./2021-10-26_13-55-13-front.mp4')
    points = [[900, 25], [1250, 25], [1250, 250], [900, 250]]
    feature_params,lk_params,color,old_gray,p0 = init_feature(cap,points)
    tracert_feature(cap,feature_params,lk_params,color,old_gray,p0,points)    
    cv2.destroyAllWindows()
    
        