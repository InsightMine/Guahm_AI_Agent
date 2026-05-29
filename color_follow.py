# RC Car Color Tracking - ESP32-CAM (ratio deadzone + proportional turn speed)
import cv2
import numpy as np
import time, requests
from threading import Thread, Lock

# --------------------
# 설정
# --------------------
ESP_IP = "192.168.1.3"
STREAM_URL = f"http://{ESP_IP}:81/stream"
ACTION_URL = f"http://{ESP_IP}/action"

# HSV 색상 범위
HSV_R1 = (np.array([0,90,70]),   np.array([10,255,255]))
HSV_R2 = (np.array([170,90,70]), np.array([179,255,255]))
HSV_G  = (np.array([35,60,60]),  np.array([85,255,255]))
HSV_B  = (np.array([100,70,60]), np.array([135,255,255]))

# 파라미터
TARGET = "R"

# [변경] 방법1: 비율 기반 데드존/미세조정 구간
CENTER_DEAD_RATIO = 0.08   # 화면 너비의 8%
CENTER_FINE_RATIO = 0.15   # 화면 너비의 15%

# [변경] 방법2: 편차 비례 회전 속도
#   - err_norm(0~1)에 따라 BASE ~ MAX 사이 선형 변화
BASE_TURN_SPEED = 70
MAX_TURN_SPEED  = 140

FORWARD_SPEED, BACKWARD_SPEED = 100, 85
CMD_COOLDOWN, MIN_AREA = 0.10, 1500
TARGET_AREA, AREA_BAND = 7000, 1500
EMA_ALPHA = 0.20

# --------------------
# 버퍼 최적화 캡처
# --------------------
class VideoStreamBufferCleaner:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.grabbed = False
        self.lock = Lock()
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
    
    def read(self):
        # 수정: NumPy 배열 호환성을 위해 is not None 사용
        with self.lock:
            return (self.grabbed, self.frame.copy() if self.frame is not None else None)
    
    def release(self):
        self.stopped = True
        time.sleep(0.1)
        try:
            self.cap.release()
        except:
            pass

# --------------------
# 모터 제어
# --------------------
last_cmd = None
last_cmd_time = 0

def send_action(param):
    try:
        requests.get(ACTION_URL, params=param, timeout=0.25)
    except:
        pass

def control_motor(action, speed=None):
    global last_cmd, last_cmd_time
    now = time.time()
    if action != last_cmd and (now - last_cmd_time) > CMD_COOLDOWN:
        if speed is not None:
            send_action({"go": action, "speed": speed})
        else:
            send_action({"go": action})
        last_cmd, last_cmd_time = action, now

# --------------------
# 마스크 생성
# --------------------
def choose_mask(hsv):
    if TARGET == "R":
        mask = cv2.bitwise_or(cv2.inRange(hsv, HSV_R1[0], HSV_R1[1]),
                              cv2.inRange(hsv, HSV_R2[0], HSV_R2[1]))
    elif TARGET == "G":
        mask = cv2.inRange(hsv, HSV_G[0], HSV_G[1])
    else:
        mask = cv2.inRange(hsv, HSV_B[0], HSV_B[1])
    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

# --------------------
# 텍스트 그리기
# --------------------
def draw_text(frame, text, pos, scale=0.5, color=(0,255,255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, 1)
    x, y = pos; pad = 5
    overlay = frame.copy()
    cv2.rectangle(overlay, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, (0,0,0), 3)
    cv2.putText(frame, text, (x, y), font, scale, color, 1)
    return frame

# --------------------
# 유틸: 비례 회전 속도 계산
# --------------------
def turn_speed_from_error(abs_err, w):
    # [변경] 편차를 0~1로 정규화 후 속도 매핑
    err_norm = min(1.0, abs_err / (w / 2.0))
    spd = int(BASE_TURN_SPEED + (MAX_TURN_SPEED - BASE_TURN_SPEED) * err_norm)
    return spd

# --------------------
# 제어 로직
# --------------------
def get_status(err_x, abs_err, area, w):
    # [변경] 프레임 크기 기준 동적 데드존
    dead_x = int(w * CENTER_DEAD_RATIO)
    fine_x = int(w * CENTER_FINE_RATIO)

    if abs_err < dead_x:
        if area < TARGET_AREA - AREA_BAND:
            control_motor("forward", FORWARD_SPEED)
            return "FORWARD", (0,255,0), dead_x, fine_x
        elif area > TARGET_AREA + AREA_BAND:
            control_motor("backward", BACKWARD_SPEED)
            return "BACKWARD", (255,128,0), dead_x, fine_x
        else:
            control_motor("stop")
            return "ALIGNED", (0,255,0), dead_x, fine_x

    # [변경] 편차 비례 회전 속도 적용
    spd = turn_speed_from_error(abs_err, w)

    if abs_err < fine_x:
        if err_x < 0:
            control_motor("left", spd)
            return "< FINE", (255,200,0), dead_x, fine_x
        else:
            control_motor("right", spd)
            return "FINE >", (255,200,0), dead_x, fine_x
    else:
        if err_x < 0:
            control_motor("left", spd)
            return "<< LEFT", (255,165,0), dead_x, fine_x
        else:
            control_motor("right", spd)
            return "RIGHT >>", (255,165,0), dead_x, fine_x

# --------------------
# 메인
# --------------------
def main():
    global TARGET
    print("Connecting...")
    cap = VideoStreamBufferCleaner(STREAM_URL).start()
    time.sleep(1.0)
    print("Ready! [R/G/B: color] [Q: quit]")
    send_action({"led": "on"})
    
    cx_ema = None
    area_ema = None
    lost_since = None
    fail = 0
    
    try:
        while True:
            ok, frame = cap.read()
            # 수정: NumPy 호환성을 위해 is None 사용
            if not ok or frame is None:
                fail += 1
                if lost_since is None:
                    lost_since = time.time()
                if time.time() - lost_since > 1.0:
                    control_motor("stop")
                if fail >= 20:
                    cap.release(); time.sleep(0.5)
                    cap = VideoStreamBufferCleaner(STREAM_URL).start()
                    fail = 0
                time.sleep(0.05)
                continue
            
            fail = 0
            h, w = frame.shape[:2]
            if max(h, w) > 720:
                frame = cv2.resize(frame, (w//2, h//2)); h, w = frame.shape[:2]
            
            frame = cv2.GaussianBlur(frame, (5,5), 0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = choose_mask(hsv)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target = None; max_area = 0
            for c in cnts:
                a = cv2.contourArea(c)
                if a > max_area:
                    max_area = a; target = c
            
            # 수정: NumPy 호환성을 위해 is not None 사용
            if target is not None and max_area > MIN_AREA:
                lost_since = None
                x, y, wb, hb = cv2.boundingRect(target)
                cx = x + wb // 2; cy = y + hb // 2
                if cx_ema is None:
                    cx_ema, area_ema = cx, max_area
                else:
                    cx_ema  = int(EMA_ALPHA * cx + (1 - EMA_ALPHA) * cx_ema)
                    area_ema = int(EMA_ALPHA * max_area + (1 - EMA_ALPHA) * area_ema)
                
                err_x  = cx_ema - (w // 2)
                abs_err = abs(err_x)

                status, color, dead_x, fine_x = get_status(err_x, abs_err, area_ema, w)
                
                # 시각화
                cv2.rectangle(frame, (x, y), (x+wb, y+hb), (0,255,0), 3)
                cv2.circle(frame, (cx_ema, cy), 8, (255,255,255), -1)
                cv2.circle(frame, (cx_ema, cy), 10, (0,255,0), 2)
                cv2.circle(frame, (w//2, h//2), 6, (0,255,255), -1)

                # [변경] 동적 라인: 비율 데드존/미세조정
                cv2.line(frame, (w//2, 0), (w//2, h), (255,255,0), 2)
                cv2.line(frame, (w//2-dead_x, 0), (w//2-dead_x, h), (0,255,0), 3)
                cv2.line(frame, (w//2+dead_x, 0), (w//2+dead_x, h), (0,255,0), 3)
                cv2.line(frame, (w//2-fine_x, 0), (w//2-fine_x, h), (255,165,0), 2)
                cv2.line(frame, (w//2+fine_x, 0), (w//2+fine_x, h), (255,165,0), 2)

                draw_text(frame, status, (10, 25), 0.5, (0,255,255))
                cn = {"R":"Red", "G":"Green", "B":"Blue"}[TARGET]
                txt = f"err:{err_x:+4d} | area:{area_ema} | {cn}"
                draw_text(frame, txt, (w - len(txt)*9 - 10, h - 15), 0.45, (200,200,200))
            
            else:
                if lost_since is None:
                    lost_since = time.time()
                control_motor("stop")
                lost_duration = time.time() - lost_since
                draw_text(frame, f"LOST ({lost_duration:.1f}s)", (10, 25), 0.5, (0,255,255))
                cn = {"R":"Red", "G":"Green", "B":"Blue"}[TARGET]
                txt = f"Track: {cn}"
                draw_text(frame, txt, (w - len(txt)*9 - 60, h - 15), 0.45, (200,200,200))
            
            vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            display = np.hstack([frame, vis_mask])
            cv2.imshow("RC Car Tracking", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                TARGET = "R"; cx_ema = area_ema = None
                print("[RED] mode"); control_motor("stop")
            elif key == ord('g'):
                TARGET = "G"; cx_ema = area_ema = None
                print("[GREEN] mode"); control_motor("stop")
            elif key == ord('b'):
                TARGET = "B"; cx_ema = area_ema = None
                print("[BLUE] mode"); control_motor("stop")
    
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        control_motor("stop"); send_action({"led":"off"})
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()