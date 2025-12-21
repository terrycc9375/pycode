"""
1. Dataset processing
	- load image from snapshot per 0.5 seconds
    - SIZE_X x SIZE_Y cropping
"""

# base packages
import os
import time

# windows api
import win32api
import win32gui
import win32ui
import PIL, PIL.Image

def get_image_from_game():
	hwnd = win32gui.FindWindow(None, "勝利女神：妮姬")
	if not hwnd:
		print("Cannot find NIKKE.")
		return None
	return hwnd

if __name__ == "__main__":
    print(get_image_from_game())




