import cv2
import numpy as np
import matplotlib.pyplot as plt

 # sisteme verdigimiz fotografların nerede oldugunu belirtip verileri okuyoruz
#img = cv2.imread("D:\Bitirme_Projesi\garbage_proje\images\\23.jpg") # kagıt
#img = cv2.imread("D:\Bitirme_Projesi\garbage_proje\images\\9.jpg")  # plastik sise
#img = cv2.imread("D:\Bitirme_Projesi\garbage_proje\images\\65.jpg") #cam sise
img = cv2.imread("D:\Bitirme_Projesi\garbage_proje\images\\59.jpg")  # poşet , resize yapmadan daha basarılı nesne tespiti yapılıyor bu veride
#img = cv2.imread("D:\Bitirme_Projesi\garbage_proje\images\\56.jpg") # fotograf boyutu cok kucuk oldugu icin bir alt satırdaki resize calıstırarak daha rahat goruntuleyebilirsizniz
#img = cv2.resize(img,(800,680))  # boyutu cok büyük veya küçük olan fotografları daha rahat goruntulemek icin yeniden boyutlandırıyoruz


img_width = img.shape[1] # sisteme verilen fotografın boyutlarını buluyoruz , numpy kutuphanesi sayesinde
img_height = img.shape[0]



img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True, crop=False) # görüntüyü algoritmaya verebilmek için blob yani 4 boyutlu sensorlere cevrilmiş hali

labels = ["garbage"] # etiketimizi sisteme veriyoruz

color = (0,255,0) # etiketin rengi ne olsun bunu belirledik



model = cv2.dnn.readNetFromDarknet("D:\Bitirme_Projesi\garbage_proje\garbage_yolov4.cfg","D:\Bitirme_Projesi\garbage_proje\garbage_yolov4_last.weights")
# cfg dosyasımızın ve egitim sonucunda elde ettigimiz weight dosyamızın nerede oldugunu belirtiyoruz

layers = model.getLayerNames() # modelimin icindeki tüm layer'ları çektik
output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()] # çıktı katmanlarını bulmalıyız -1 yapmamızın sebebi yolo katmanlarındaki degerlerin cıktı katmanındaki degerlere eşitleyebilmek

model.setInput(img_blob) # blob yaptıgımız tensörü modelimize veriyoruz .

detection_layers = model.forward(output_layer) # layer larımız icinden tespit yapılanları alıyoruz

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        
        scores = object_detection[5:] # ilk 5 deger bouding box ile ilgili biz bundan sonrakileri alıcaz
        predicted_id = np.argmax(scores) # scores tuttugu degerler icinden max indexi bulacagız
        confidence = scores[predicted_id] # güven skoruna erişecegiz yani kısaca bir nesnenin tespitinde benzetmeinde de tespit yapacaktır biz ise en cok güvenebilecegimiz degere yani score a erisecegiz
        
        if confidence > 0.20:  #nesne tespitinde yuzde kac basarıda nesnenin çöp olarak belirtilmesi gerektigini belirliyoruz
            
            label = labels[0] # bir tane etiketimiz oldugu icin direkt labels[0] dedik
            bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height]) # en ve boy ile bounding_box ı genişlettik
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int") # bounding box ta gelen degeri in e cevirecegiz bunlar float cunku
            
            start_x = int(box_center_x - (box_width/2))  # diktörtgenimizin baslangıc ve bitiş noktalarını belirledik
            start_y = int(box_center_y - (box_height/2))
            
            end_x = start_x + box_width # baslangıca boyutu eklersek bitişni bulmus oluruz
            end_y = start_y + box_height
            
            box_color = color # yukarıda label ımız icin renk belirlemiştik
            
            
            
            label = "{}: {:.2f}%".format(label, confidence*100) # label degerlerimizi de ekrana yazdıralım
            print("object {}".format(label))
            print("    box_height :"+str(box_height)+"   box_width:"+str(box_width)+"  ")
            cv2.rectangle(img, (start_x,start_y),(end_x,end_y),box_color,1)  # tespit edilen nesneyi bir dikdörtgen içerisinde gösteririz
            cv2.putText(img,label,(start_x,start_y - 10 ), cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)

cv2.imshow("Detection Window", img) #nesne tespiti yapılmıs görüntüyü ekrana yansıtıyoruz
cv2.waitKey(0)
cv2.destroyAllWindows()
        


        
    




















