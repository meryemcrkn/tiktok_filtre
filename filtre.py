import cv2
import numpy as np

# Haarcascade modellerini yükleyelim (Yüz, göz ve ağız tespiti için hazır modeller)
yuz_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
goz_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
gul_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Kamerayı başlat
kamera = cv2.VideoCapture(0)

while True:
    # Kameradan görüntü al
    ret, kare = kamera.read()
    if not ret:
        break

    # Görüntüyü gri tona çevir (Haarcascade için gereklidir)
    gri_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    yuzler = yuz_cascade.detectMultiScale(gri_kare, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

    for (x, y, w, h) in yuzler:
        # Yüz bölgesini ayır
        roi_renkli = kare[y:y + h, x:x + w]
        gri_yuz = gri_kare[y:y + h, x:x + w]

        # Gözleri tespit et
        gozler = goz_cascade.detectMultiScale(gri_yuz, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        for (ex, ey, ew, eh) in gozler:
            if ey < h // 2:  # Gözlerin yüzün üst kısmında olduğundan emin ol
                goz_roi = roi_renkli[ey:ey + eh, ex:ex + ew]  # Göz bölgesini al
                goz_roi = cv2.flip(goz_roi, 0)  # Gözleri dikey eksende çevir
                roi_renkli[ey:ey + eh, ex:ex + ew] = goz_roi  # Değiştirilen gözleri yüz bölgesine yerleştir

        # Ağız tespiti
        agizlar = gul_cascade.detectMultiScale(gri_yuz, scaleFactor=1.3, minNeighbors=15, minSize=(30, 30))
        for (sx, sy, sw, sh) in agizlar:
            if sy > h // 2:  # Ağız yüzün alt kısmında olmalı
                agiz_roi = roi_renkli[sy:sy + sh, sx:sx + sw]  # Ağız bölgesini al
                agiz_roi = cv2.flip(agiz_roi, 0)  # Ağızı dikey eksende çevir
                roi_renkli[sy:sy + sh, sx:sx + sw] = agiz_roi  # Değiştirilen ağızı yüz bölgesine yerleştir

    # Filtrelenmiş görüntüyü ekranda göster
    cv2.imshow('Ters Dünya', kare)

    # 'q' tuşuna basıldığında çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat ve pencereleri kapat
kamera.release()
cv2.destroyAllWindows()
