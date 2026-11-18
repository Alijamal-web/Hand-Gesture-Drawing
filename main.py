import cv2
import mediapipe as mp
import numpy as np

# Камера
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Полотно для рисования
canvas = None

# Предыдущая позиция пальца
prev_x, prev_y = None, None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera does not work")
        break

    # Зеркальное изображение
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            # Координаты пальцев
            h, w, _ = frame.shape

            index_tip = hand.landmark[8]
            middle_tip = hand.landmark[12]

            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            middle_x = int(middle_tip.x * w)
            middle_y = int(middle_tip.y * h)

            # Проверяем, подняты ли указательный и средний пальцы
            index_up = hand.landmark[8].y < hand.landmark[6].y
            middle_up = hand.landmark[12].y < hand.landmark[10].y

            # ДВА ПАЛЬЦА = СТИРАТЕЛЬ
            if index_up and middle_up:

                cv2.circle(
                    frame,
                    (index_x, index_y),
                    40,
                    (0, 0, 255),
                    2
                )

                cv2.circle(
                    canvas,
                    (index_x, index_y),
                    40,
                    (0, 0, 0),
                    -1
                )

                prev_x, prev_y = None, None

            # ОДИН ПАЛЕЦ = РИСОВАНИЕ
            elif index_up:

                cv2.circle(
                    frame,
                    (index_x, index_y),
                    8,
                    (0, 255, 0),
                    -1
                )

                if prev_x is not None and prev_y is not None:
                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (index_x, index_y),
                        (255, 0, 255),
                        5
                    )

                prev_x, prev_y = index_x, index_y

            # Ничего не делаем
            else:
                prev_x, prev_y = None, None

            # Показываем точки руки
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

    # Объединяем камеру и рисунок
    frame = cv2.add(frame, canvas)

    cv2.imshow("Finger Drawing", frame)

    # Q = выход
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

