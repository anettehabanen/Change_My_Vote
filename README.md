# Change_My_Vote

This is the Reflex application for my master thesis called "Deep Fakes for Paper Vote Privacy Defence". The main project can be found here: https://github.com/anettehabanen/Deep_Fakes_for_Paper_Vote_Privacy_Defence

### Clone the repository:
```
git clone https://github.com/anettehabanen/Change_My_Vote
cd Change_My_Vote
```

### Create Python environment

NB! This pipeline is tested for Python versions 3.10-3.12. To check the Python version:
```
python3 --version
```

Creating an environment and installing requirements.
```
python3 -m venv reflex_venv   # Create an environment
source reflex_venv/bin/activate   # Activate the environment
pip install -r requirements.txt   # Install the requirements
```

### Run the project:
```
reflex run   # or 'reflex run --env dev'
```

The webpage can be found at http://localhost:3001/

If the port 8001 or 3001 is already in use, go to the [config file](rxconfig.py) and change them to open ports.


## Example usage

### Step 1: input data

In the first page you need to upload a video of a voting ballot and an image of 3-digit code. For this example, upload the image [per_4_img_2.png](example_files/per_4_img_2.png) and video [per_6_vid_4_filled.mp4](example_files/per_6_vid_4_filled.mp4) from the 'example_files' subfolder. Then click 'Start configuring'.

### Step 2: crop 3-digit code
On this page you need to crop out the 3-digit code from the image. To do this set the range for x and y coordinates where the digits are. 

1. Click 'Update image' to see the uploaded image.
2. Then insert the below-mentioned numbers on the sliders (the numbers do not have to be exact).
3. Click 'Crop image' to see the cropped image of the 3-digit code.
4. Click 'Next' to move to the next step.

For the image 'per_4_img_2.png' the numbers are:
* x-coordinates: from 2150 to 2700
* y-coordinates: from 1400 to 1700


### Step 3: configure corners
In this step you need to set the box corner coordinates for the first frame.

1. Click 'Update frame' to see the first frame of the uploaded video.
2. Then insert the above-mentioned numbers to the text-fields.
3. Click 'Craw points' to see where the points are located in the frame.
4. Click 'Next' to move to the final step.

For the video 'per_6_vid_4_filled.png' the numbers are:
* Point A: x = 1017, y = 417
* Point B: x = 1263, y = 407
* Point C: x = 1023, y = 507
* Point D: x = 1272, y = 497

### Step 4: results
In the final step press the 'Start converting' button and wait for the pipeline to create the changed video. When the pipeline is finished, it will show the new video on the page. 

NB! This pipeline works correctly if you have GPU.

## Example videos
The folder 'example_videos' has some images and videos of ballots and digits. To test the Reflex app, use these files when asked to upload the image of 3-digit code and a video of a voting ballot. Testing with your own videos/images is also possible.
