import math

# main function for detection, thumb prototype still not working
def detect_gesture(hand):

    index_up = hand[8].y < hand[6].y
    middle_up = hand[12].y < hand[10].y
    ring_up = hand[16].y < hand[14].y
    pinky_up = hand[20].y < hand[18].y

    thumb_tip = hand[4]
    thumb_base = hand[2]

    thumb_distance = math.sqrt(
        (thumb_tip.x - thumb_base.x) ** 2 +
        (thumb_tip.y - thumb_base.y) ** 2
    )

    thumb_up = thumb_distance > 0.10

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "POINT"

    elif index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"

    elif index_up and middle_up and ring_up and pinky_up:
        return "OPEN PALM"

    elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "THUMBS UP"

    elif not index_up and not middle_up and not ring_up and not pinky_up:
        return "FIST"

    return "UNKNOWN"