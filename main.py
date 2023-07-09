# !/usr/bin/env python
"""
@File    : main.py
@Time    : 2023/07/08 23:59:59
@Author  : Yizhou Liu
@Version : 0.1
@Contact : liu_yizhou@outlook.com
@Desc    : A simple tool to extract slides from videos.
"""

import os
import sys

import cv2
import img2pdf
from moviepy.editor import VideoFileClip
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def sanitize_path(path):
    special_chars = [
        "<",
        ">",
        ":",
        '"',
        "/",
        "\\",
        "|",
        "?",
        "*",
        "丨",
        "：",
        " ",
        "-",
    ]  # Add or remove special characters as needed
    for char in special_chars:
        path = path.replace(char, "_")
    return path


def extract_frames(directory, video, seconds_interval, crop_points, resize_dim):
    video_name = sanitize_path(os.path.splitext(video)[0])
    clip = VideoFileClip(os.path.join(directory, video))
    duration = int(clip.duration)

    os.makedirs(os.path.join(directory, "audio"), exist_ok=True)
    os.makedirs(os.path.join(directory, "slides"), exist_ok=True)

    # Extract audio
    audio = clip.audio
    audio.write_audiofile(os.path.join(directory, "audio", f"{video_name}.mp3"))

    previous_image = None
    imgs = []
    for i in tqdm(range(0, duration, seconds_interval)):
        frame_img = clip.get_frame(i)
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)

        # Crop the image
        cropped_img = frame_img[
            crop_points[0] : crop_points[1], crop_points[2] : crop_points[3]
        ]

        # Resize the image
        resized_img = cv2.resize(cropped_img, resize_dim)

        # Compare with the previous image
        if previous_image is not None:
            grayA = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

            (score, _) = ssim(grayA, grayB, full=True)
            if score < 0.95:
                img_path = os.path.join(directory, "slides", f"{i:0>4}.jpg")
                cv2.imencode(".jpg", resized_img)[1].tofile(img_path)
                imgs.append(img_path)

        previous_image = resized_img

    # After processing all frames, convert images to PDF
    with open(os.path.join(directory, "slides", f"{video_name}.pdf"), "wb") as f:
        f.write(img2pdf.convert(imgs))

    # Delete images
    for img_path in imgs:
        os.remove(img_path)

    clip.close()


if __name__ == "__main__":
    # Define your crop points (y1, y2, x1, x2)
    CROP_POINTS = [68, 681, 72, 1223]
    # Define your resize dimension (height)
    HEIGHT = 900

    RATIO = (CROP_POINTS[3] - CROP_POINTS[2]) / (CROP_POINTS[1] - CROP_POINTS[0])
    RESIZE_DIM = (int(HEIGHT * RATIO), HEIGHT)

    DIRECTORY = sys.argv[1]
    file_list = [
        x for x in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, x))
    ]
    for file in tqdm(file_list):
        extract_frames(DIRECTORY, file, 60, CROP_POINTS, RESIZE_DIM)
