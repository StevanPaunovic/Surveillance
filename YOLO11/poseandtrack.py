from ultralytics import YOLO
import cv2

# Modelle laden
posemodel = YOLO("yolo11n-pose.pt")  # YOLO-Pose-Modell
trackmodel = YOLO("yolo11l.pt")  # YOLO-Tracking-Modell

# Kamera starten
cap = cv2.VideoCapture(0)  # Verwende Kamera 0
if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler: Kein Bild von der Kamera.")
        break

    # Pose-Schätzung durchführen
    poseresults = posemodel(frame)
    for result in poseresults:
        for box in result.boxes:
            if hasattr(result, "keypoints"):
                keypoints = result.keypoints.cpu().numpy()  # Extrahiere Keypoints
                for kp in keypoints:
                    # Überprüfen, ob die Keypoints eine Liste von Koordinaten sind
                    if len(kp) >= 2:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Tracking durchführen
    trackresults = trackmodel.track(frame)
    for result in trackresults:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            track_id = box.id
            cls = int(box.cls[0])
            label = f"ID: {track_id}, Class: {trackmodel.names[cls]}"
            # Bounding Box und Label zeichnen
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Ergebnisse anzeigen
    cv2.imshow("YOLO Pose + Tracking", frame)

    # Beenden mit Taste 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera und Fenster freigeben
cap.release()
cv2.destroyAllWindows()
