
import reflex as rx
from rxconfig import config
from style import main_color
import os



class InputImageState(rx.State):

    # The images to show.
    img: list[str] = ["default-image.jpg"]
    img_uploaded: bool = False

    async def handle_image_upload(self, files: list[rx.UploadFile]):

        if self.img[0] != "default-image.jpg":
            path = "{}/uploaded_files/{}".format(os.getcwd(), self.img[0])
            self.delete_file(path)

        for file in files:
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename

            # Save the file.
            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

            # Update the img var.
            self.img[0] = file.filename
            self.img_uploaded = True

    def delete_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)


    def clear_image(self):
        path = "{}/uploaded_files/{}".format(os.getcwd(), self.img[0])
        if os.path.exists(path):
            self.delete_file(path)
            self.img[0] = "default-image.jpg"
            self.img_uploaded = False




def input_image():
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
                id="image_upload",
                max_files=1,
                border=f"1px dotted {main_color}",
                padding="2em",
            ),

            rx.hstack(
                rx.text("Selected file:"),
                rx.foreach(
                    rx.selected_files("image_upload"), rx.text
                ),
                padding_top="20px"
            ),
            rx.hstack(
                rx.button(
                    "Upload",
                    class_name="dark_button",
                    on_click=InputImageState.handle_image_upload(
                        rx.upload_files(upload_id="image_upload")
                    ),
                ),
                rx.button(
                    "Clear",
                    class_name="dark_button",
                    on_click=InputImageState.clear_image(), #on_click=rx.clear_selected_files("image_upload"),
                ),
                padding_top="20px"
            ),
            border=f"1px dotted {main_color}",
            padding="1em",
        ),
        rx.cond(
            InputImageState.img_uploaded,
            rx.foreach(
                InputImageState.img,
                lambda img: rx.image(
                    src=rx.get_upload_url(img),
                    width="400px",
                    height="auto"
                ),
            ),
            rx.image(
                src="default-image.jpg",
                width="400px",
                height="auto"
            ),
        ),
        padding="0.5em",
     )