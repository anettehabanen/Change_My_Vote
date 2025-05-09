import reflex as rx
from reflex.vars import Var
from rxconfig import config

import time
import numpy as np
import os
from ultralytics import YOLO
from moviepy import VideoFileClip

from .pipeline import save_video_as_frames
from .pipeline import tracking_with_bounding_boxes
from .pipeline import creating_missing_bboxes
from .pipeline import changing_digits_in_the_frames
from .pipeline import lucas_kanade_video
from .pipeline import prepare_new_code_image
from .pipeline import creating_video_from_frames
from .pipeline import getting_video_audio
from .pipeline import adding_audio_to_a_video
from .pipeline import remove_files

from .input_video import InputVideoState
from .input_image import InputImageState
from .configure_new_code import xSliderState
from .configure_new_code import ySliderState
from .configure_corners import SlidersState




path = "{}/pipeline_files".format(os.getcwd())
video_changed_fps = "{}/fps_".format(path)
frames_path = "{}/frames".format(path) # video frames
yolo_boxes_path = "{}/yolo".format(path)
bboxes_path = "{}/labels".format(path)
square_digits_path = "{}/square".format(path) # bounding boxes
changed_frames_path = "{}/frames_changed".format(path) # video frames
tracked_labels_path = "{}/corners_labels".format(path) # frame corner labels
audio = "{}/audio.mp3".format(path) # audio file
video_frames = "{}/new_frames.mp4".format(path) # video without audio
video_final = "{}/assets".format(os.getcwd()) # video with audio

yolo_model = YOLO("{}/model_weights/YOLO_10_+e200_best.pt".format(os.getcwd()))

wavepaint_predict = "{}/Wavepaint/predict.py".format(os.getcwd())
wavepaint_weights = "{}/model_weights/WavePaint_blocks4_dim128_modules6_bg.pth".format(os.getcwd())

id_length = 6
padding_up_down = 20 # How many pixels we add beyond bounding box pixels (20 means 10 pixels for each side)
padding_sides = 20 
add_to_boxes_height = 20 # saving little bit bigger boxes than yolo predicts
add_to_boxes_width = 7 # saving little bit bigger boxes than yolo predicts




steps = ["Part 1: saving video as frames",
         "Part 2: tracking corners",
         "Part 3: getting YOLO predictions",
         "Part 4: getting and placing Wavepaint predictions, placing new digit",
         "Part 5: creating new video"]

progress_values = [10, 10, 20, 50, 10]



class ResultState(rx.State):
    step_name: str = ""
    progress_value: int = 0
    done: str = ""
    video: str = ""
    image: str = ""
    result_video: str = ""
    digit_xmin: int = 0
    digit_xmax: int = 0
    digit_ymin: int = 0
    digit_ymax: int = 0
    corner_Ax: int = 0
    corner_Ay: int = 0
    corner_Bx: int = 0
    corner_By: int = 0
    corner_Cx: int = 0
    corner_Cy: int = 0
    corner_Dx: int = 0
    corner_Dy: int = 0
    show_progres: bool = False
    show_video: bool = False
    lower_fps: int = 24

    @rx.background
    async def converting(self, video_path, image_path, xmin, xmax, ymin, ymax, Ax, Ay, Bx, By, Cx, Cy, Dx, Dy):
        async with self:
            self.step_name = steps[0]
            self.progress_value = 0
            self.done = ""
            self.video = video_path[0]
            self.image = image_path[0]
            self.digit_xmin = xmin
            self.digit_xmax = xmax
            self.digit_ymin = ymin
            self.digit_ymax = ymax
            self.corner_Ax = Ax
            self.corner_Ay = Ay
            self.corner_Bx = Bx
            self.corner_By = By
            self.corner_Cx = Cx
            self.corner_Cy = Cy
            self.corner_Dx = Dx
            self.corner_Dy = Dy
            self.show_video = False
            self.show_progres = True

        step1_result = self.step1()
        if step1_result:
            async with self:
                self.step_name = steps[1]
                self.progress_value += progress_values[0]

        step2_result = self.step2()
        if step2_result:
            async with self:
                self.step_name = steps[2]
                self.progress_value += progress_values[1]

        step3_result = self.step3()
        if step3_result:
            async with self:
                self.step_name = steps[3]
                self.progress_value += progress_values[2]

        step4_result = self.step4()
        if step4_result:
            async with self:
                self.step_name = steps[4]
                self.progress_value += progress_values[3]

        step5_result = self.step5()
        if len(step5_result) != 0 :
            async with self:
                self.progress_value += progress_values[4]
                self.done = "Here is the new video:"
                self.result_video = "/final_result.mp4"
                self.show_video = True



    def step1(self):
        print("Starting step 1")

        if not os.path.exists(path):
            os.makedirs(path)

        video_path = "{}/uploaded_files/{}".format(os.getcwd(), self.video)
        video_fps = VideoFileClip(video_path)
        video_path_changed_fps = "{}{}".format(video_changed_fps, self.video)
        video_fps.write_videofile(video_path_changed_fps, fps=self.lower_fps)
        save_video_as_frames(video_path_changed_fps, frames_path, id_length)
        return True
    
    def step2(self):
        print("Starting step 2")

        if not os.path.exists(tracked_labels_path):
            os.makedirs(tracked_labels_path)
        
        features = np.float32(np.array([[[self.corner_Ax, self.corner_Ay]], 
                                        [[self.corner_Bx, self.corner_By]], 
                                        [[self.corner_Cx, self.corner_Cy]], 
                                        [[self.corner_Dx, self.corner_Dy]]]))
        lucas_kanade_video(frames_path, tracked_labels_path, features)
        return True

    def step3(self):
        print("Starting step 3")
        first_3_frame_idx = tracking_with_bounding_boxes(yolo_model, frames_path, yolo_boxes_path, tracked_labels_path, add_to_boxes_width, add_to_boxes_height)
        creating_missing_bboxes(yolo_boxes_path, bboxes_path, first_3_frame_idx, id_length)
        return True
    
    def step4(self):
        print("Starting step 4")

        image_path = "{}/uploaded_files/{}".format(os.getcwd(), self.image)
        new_digit = prepare_new_code_image(image_path, 
                                           self.digit_xmin, 
                                           self.digit_xmax, 
                                           self.digit_ymin, 
                                           self.digit_ymax)




        changing_digits_in_the_frames(frames_path, 
                                      bboxes_path, 
                                      square_digits_path, 
                                      changed_frames_path, 
                                      id_length, 
                                      padding_up_down, 
                                      padding_sides, 
                                      wavepaint_predict, 
                                      wavepaint_weights,
                                      tracked_labels_path,
                                      new_digit)
        return True


    def step5(self):
        print("Starting step 5")
        video_path_changed_fps = "{}{}".format(video_changed_fps, self.video)
        creating_video_from_frames(changed_frames_path, video_frames, self.lower_fps)
        getting_video_audio(video_path_changed_fps, audio)
        result_video_path = "{}/final_result.mp4".format(video_final)
        adding_audio_to_a_video(video_frames, audio, result_video_path)

        remove_files(path)
        return result_video_path




def results(): 
    return rx.container(
        rx.center(
            rx.vstack(
                rx.button("Start converting", 
                        class_name="dark_button",
                        on_click=ResultState.converting(InputVideoState.vid,
                                                        InputImageState.img,
                                                        xSliderState.x_start,
                                                        xSliderState.x_end,
                                                        ySliderState.y_start,
                                                        ySliderState.y_end,
                                                        SlidersState.Ax,
                                                        SlidersState.Ay,
                                                        SlidersState.Bx,
                                                        SlidersState.By,
                                                        SlidersState.Cx,
                                                        SlidersState.Cy,
                                                        SlidersState.Dx,
                                                        SlidersState.Dy),
                        disabled=ResultState.show_progres,
                ),
                rx.cond(
                    (ResultState.show_progres) & (ResultState.done == "") ,
                    rx.hstack(
                        rx.spinner(size="3"),
                        rx.heading(ResultState.step_name),
                    )
                ),
                rx.cond(
                    (ResultState.show_progres) & (ResultState.done == ""),
                    rx.progress(value=ResultState.progress_value, 
                                max=100,
                                color_scheme="purple"
                    ),
                ),

                rx.cond(
                    ResultState.show_video,
                    rx.heading(ResultState.done),
                ),

                rx.cond(
                    ResultState.show_video,
                    rx.video(
                        url=ResultState.result_video,
                        width="auto",
                        height="auto",
                    ),
                ),

                width="100%"

            ),
        ),
    )

