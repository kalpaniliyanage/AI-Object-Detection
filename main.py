from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import queue
import time

# --- Setup ---
# Reverted to plain yolov8s.pt on purpose: YOLO-World requires loading a CLIP
# TorchScript archive, which is crashing on this machine with a low-level
# "bad allocation" error during torch.jit.load — that's a torch/Windows
# memory issue unrelated to your script, and not worth fighting right now.
# This model only detects COCO's 80 classes (see list below), but it is
# guaranteed to load and run without that crash.
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(0)

# --- Object Dictionary ---
# NOTE: only the following keys are COCO classes yolov8s.pt can actually
# detect: person, bottle, cup, book, cell phone, laptop, mouse, keyboard,
# scissors. The rest (pen, pencil, eraser, ruler, marker, notebook,
# calculator, stapler, paper) are kept here for later — once you're ready,
# we train a small custom model for these specifically (separate task).
object_info = {
    "person": "I have detected a person. Humans can think and perform tasks.",
    "bottle": "I have detected a bottle. Bottles are commonly used to store drinking water and other liquids.",
    "cup": "I have detected a cup. Cups are used for drinking.",
    "book": "I have detected a book. Books contain knowledge and stories.",
    "cell phone": "I have detected a mobile phone. It is used for communication.",
    "laptop": "I have detected a laptop. It is a portable computer.",
    "mouse": "I have detected a mouse. It controls the computer cursor.",
    "keyboard": "I have detected a keyboard. It is used for typing.",
    "scissors": "I have detected scissors. They are used for cutting.",
    "pen": "I have detected a pen. A pen is used for writing with ink.",
    "pencil": "I have detected a pencil. A pencil is used for writing and drawing.",
    "eraser": "I have detected an eraser. It removes pencil marks.",
    "ruler": "I have detected a ruler. It measures length.",
    "marker": "I have detected a marker. It is used for writing.",
    "notebook": "I have detected a notebook. It is used for taking notes.",
    "calculator": "I have detected a calculator. It performs calculations.",
    "stapler": "I have detected a stapler. It joins papers together.",
    "paper": "I have detected paper. It is used for writing and printing.",
}

# --- Thread-safe speech queue ---
# NOTE: On Windows, pyttsx3's SAPI5 engine has a well-known bug where reusing
# one engine instance across multiple say()/runAndWait() calls works for the
# FIRST utterance only, then silently stops responding — which is exactly
# what you were seeing (first object announced, nothing after). The fix is
# to create a fresh engine for every single message.
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak(text):
    # Drop any message still waiting to be spoken. Without this, if objects
    # change faster than speech keeps up, the voice ends up playing stale,
    # outdated announcements that no longer match what's on screen. This
    # guarantees the voice always catches up to the LATEST detection.
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            break
    speech_queue.put(text)

# --- Re-announcement cooldown ---
# Every detected object keeps re-announcing with sound at this interval for
# as long as it stays in view. When NOTHING is detected, a separate "no
# object" message also keeps announcing with sound at its own interval, so
# the system is always giving spoken feedback either way.
COOLDOWN_SECONDS = 4.0
NO_OBJECT_COOLDOWN_SECONDS = 4.0
last_announced = {}
last_no_object_announced = 0
NO_OBJECT_MESSAGE = "No object detected. Please show an object to the camera."

print("System Active (COCO classes only). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.6, verbose=False)

    current_frame_objects = set()
    for result in results:
        for box in result.boxes:
            name = model.names[int(box.cls[0])]
            current_frame_objects.add(name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    now = time.time()
    if current_frame_objects:
        for obj in current_frame_objects:
            last_time = last_announced.get(obj, 0)
            if now - last_time >= COOLDOWN_SECONDS:
                # Every detected object gets a spoken message, no exceptions.
                # If it's in our dictionary, use the custom message. If it's
                # a COCO object we haven't written a custom line for, still
                # announce it by name so it's never silently skipped.
                if obj in object_info:
                    message = object_info[obj]
                else:
                    message = f"I have detected an object, it looks like a {obj}, but I don't have detailed information about it."
                speak(message)
                last_announced[obj] = now
    else:
        # Nothing detected this frame — keep giving spoken feedback instead
        # of going silent, repeating at NO_OBJECT_COOLDOWN_SECONDS.
        if now - last_no_object_announced >= NO_OBJECT_COOLDOWN_SECONDS:
            speak(NO_OBJECT_MESSAGE)
            last_no_object_announced = now
        last_announced.clear()  # reset so objects re-announce fresh next time they appear

    # --- On-screen "understanding" panel ---
    panel_y = 25
    overlay = frame.copy()
    panel_height = 30 + 22 * max(len(current_frame_objects), 1)
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    if current_frame_objects:
        for obj in current_frame_objects:
            message = object_info.get(obj, f"I have detected a {obj}.")
            cv2.putText(frame, message, (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            panel_y += 22
    else:
        cv2.putText(frame, "No objects detected", (10, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("AI Object Narrator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)