import numpy as np
import cv2

def init_feature(cap): #初始化特徵點
    #設定特徵取樣參數
    feature_params = dict( maxCorners = 100, qualityLevel =0.3, minDistance =7, blockSize =7)
    lk_params = dict(winSize = (15,15), maxLevel = 5, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
    color = np.random.randint(0,255,(100,3)) #隨機色彩
    
    ret, old_frame = cap.read()  #讀取影片
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) #轉灰階
    
    
    x,y= old_gray.shape #取得frame大小
    
    
    mask1 = np.zeros([x, y], dtype=np.uint8) #新增遮罩陣列
    mask1[50:300, 600:y-50] = 255 #設定特徵取樣範圍
    
    #cv2.imshow("mask1", mask1) #顯示遮罩
    #cv2.waitKey(0)

    old_gray1 = cv2.add(old_gray, np.zeros(np.shape(old_gray), dtype=np.uint8), mask=mask1)
    cv2.imshow("image", old_gray1) #顯示遮罩後照片
    #cv2.waitKey(0)
    
    p0 = cv2.goodFeaturesToTrack(old_gray1, mask =None, **feature_params) #取特徵點
    #print(p0)
    mask = np.zeros_like(old_frame)
    #cap.release()
    
    return feature_params,lk_params,color,old_gray,p0,mask


def tracert_feature(cap,feature_params,lk_params,color,old_gray,p0,mask):
    count = 0
   
    
    while(1):
        
        
        ret,frame = cap.read()
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if not st is None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        if st is None: #特徵點已移出畫面, 重新擷取特徵點
            feature_params,lk_params,color,old_gray,p0,mask = init_feature(cap)            
            tracert_feature(cap,feature_params,lk_params,color,old_gray,p0,mask)
        
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

        img = cv2.add(frame,mask) #顯示特徵移動軌跡
        
        if (x_mean) > 0.05 and not st is None:
            print('Move forward')
            cv2.putText(img,'Move forward',(500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
        
        if (y_mean) < -3.9  and (x_mean) < 0.05 and not st is None:
            print('Turn left')
            cv2.putText(img,'Move forward',(500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
            cv2.putText(img,'Turn left',(500, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
            print('y value:'+str(y_mean))

        if (y_mean) > -2.5 and (x_mean) < 0.05 and not st is None:
            print('Turn right')
            cv2.putText(img,'Move forward',(500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
            cv2.putText(img,'Turn right',(500, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
            print('y value:'+str(y_mean))
        
        if (x_mean) < 0.01 and x_mean > -0.01 and (y_mean) < 0.01 and not st is None and (x_mean) != 0:
            print('Stop Move')
            cv2.putText(img,'Stop Move',(500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
            print('x value:'+str(x_mean))
        cv2.putText(img,'x value:'+str((x_mean)) ,(500, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)
        cv2.putText(img,'y value:'+str((y_mean)) ,(500, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA,)

            
        
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k ==27:
            break
        
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        count +=1


if __name__ == '__main__':
    
    cap = cv2.VideoCapture('./2021-10-26_14-00-14-front.mp4')
    feature_params,lk_params,color,old_gray,p0,mask = init_feature(cap)
    tracert_feature(cap,feature_params,lk_params,color,old_gray,p0,mask)    
    cv2.destroyAllWindows()
    
        