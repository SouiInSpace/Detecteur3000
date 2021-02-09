import cv2
import numpy as np
import time


captureVideo=cv2.VideoCapture("Autoroutea.mp4")
car_detect = cv2.CascadeClassifier("./cars.xml")

#Debut mesure du temps
temps_debut = time.time()

while captureVideo.isOpened():

    retour, image = captureVideo.read()
    if retour is False:
        quit()

    #conversion (niveaux de gris) et filtrage (canny) d'une zone de l'image
    zoneGris=cv2.cvtColor(image[700:1050, 710:1150], cv2.COLOR_BGR2GRAY)
    arrayZoneDetection=cv2.Canny(zoneGris, 75, 150)
    zoneAtraiter=image[400:750, 800:1055]
    #sur la vidéo la zone est matérialisée sous forme d'un rectangle rouge

    #détection obstacle
    car_detect_coordonnees = car_detect.detectMultiScale(zoneAtraiter, scaleFactor=1.1, minNeighbors=4)
    #rectangle de détection
    cv2.rectangle(image, (850, 400), (1055, 702), (0, 255, 255), 1)

    #afficher le rectanlge
    for (x, y, w, h) in car_detect_coordonnees:
        cv2.rectangle(zoneAtraiter, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, "OBSTACLE !!", (850, 350), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    cv2.rectangle(image, (1050, 700), (1150, 710), (0, 0, 255), 1)
    #balayage des pixels de la zone de détection:


    #Détection changement de ligne

    s1=-1
    zoneDetection = arrayZoneDetection[0]

    for i in range(len(zoneDetection)):    # si pixel différent de 0 c'est qu'il est blanc: a détecté début de la ligne blanche
        if zoneDetection[i]!=0:
            s1=i


    if s1!=-1:          # écrire sur l'image en fonction de l'emplacement du début de la ligne blanche
        cv2.circle(image, (1050+s1, 700), 3, (0, 255, 0), 3)
        if s1<30:
            cv2.putText(image, "!GAUCHE!", (1000, 700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)


        elif s1>60:
            cv2.putText(image, "!DROITE!", (1000, 700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        else:
            cv2.putText(image, "OK", (1000, 700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)


#Fin mesure du temps


    cv2.imshow("image", image)


    if cv2.waitKey(1) == ord('q'):
        break


captureVideo.release()
cv2.destroyAllWindows()

temps_fin = time.time()
temps = temps_fin - temps_debut
print(temps)
