import reflex as rx
from rxconfig import config
from style import main_color
import os


class InputVideoState(rx.State):

    # The images to show.
    vid: list[str] = [""]
    vid_uploaded: bool = False

    async def handle_video_upload(self, files: list[rx.UploadFile]):

        if self.vid[0] != "":
            path = "{}/uploaded_files/{}".format(os.getcwd(), self.vid[0])
            self.delete_file(path)

        for file in files:
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename

            # Save the file.
            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

            # Update the img var.
            self.vid[0] = file.filename
            self.vid_uploaded = True

    def delete_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)


    def clear_video(self):
        path = "{}/uploaded_files/{}".format(os.getcwd(), self.vid[0])
        if os.path.exists(path):
            self.delete_file(path)
            self.vid[0] = ""
            self.vid_uploaded = False



def input_video():
     return rx.container(
        rx.container(
            rx.upload(
                rx.vstack(
                    rx.text(
                        "Drag and drop or select files (max. 1 file)"
                    ),
                    rx.button(
                        "Select File",
                        class_name="light_button",
                    ),
                ),
                id="video_upload",
                max_files=1,
                border=f"1px dotted {main_color}",
                padding="2em",
            ),
            rx.hstack(
                rx.text("Selected file:"),
                rx.foreach(
                    rx.selected_files("video_upload"), rx.text
                ),
                padding_top="20px"
            ),
            rx.hstack(
                rx.button(
                    "Upload",
                    class_name="dark_button",
                    on_click=InputVideoState.handle_video_upload(
                        rx.upload_files(upload_id="video_upload")
                    ),
                ),
                rx.button(
                    "Clear",
                    class_name="dark_button",
                    on_click=InputVideoState.clear_video(), #rx.clear_selected_files("video_upload"),
                ),
                padding_top="20px"
            ),
            border=f"1px dotted {main_color}",
            padding="1em",
        ),
        rx.foreach(
            InputVideoState.vid,
            lambda vid: rx.video(
                url=rx.get_upload_url(vid),
                width="400px",
                height="auto"
            ),
        ),
        padding="0.5em",
     )