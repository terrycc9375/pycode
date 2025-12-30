"""
1. Dataset processing
	- load image from snapshot per 0.5 seconds
    - SIZE_X x SIZE_Y cropping
"""

# base packages
import os
import time

# windows api
import win32gui
import win32con

# image processing
import PIL, PIL.Image
import mss

# global constants
SIZE_X = 10
SIZE_Y = 16

def get_image_from_game():
	hwnd = win32gui.FindWindow(None, "勝利女神：妮姬")
	rect = None
	if not hwnd:
		print("Cannot find NIKKE.")
	elif win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowPlacement(hwnd)[1] != win32con.SW_SHOWMINIMIZED:
		rect = win32gui.GetWindowRect(hwnd)
	return hwnd, rect

def capture_window():
	hwnd, rect = get_image_from_game()
	if not hwnd or not rect:
		return None
	with mss.mss() as sct:
		monitor = {
			"top": rect[1],
			"left": rect[0],
			"width": rect[2] - rect[0],
			"height": rect[3] - rect[1],
		}
		screenshot = sct.grab(monitor)
		screenshot = PIL.Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
		screenshot.save("./temp/screenshot.png")
	return screenshot


if __name__ == "__main__":
	while True:
		try:
			img = capture_window()
			if img:
				print("Successed")
			else:
				print("Failed")
		except KeyboardInterrupt:
			print("Exit.")
		finally:
			time.sleep(1)




