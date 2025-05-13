# Change_My_Vote

This is the Reflex application for my master thesis called "Deep Fakes for Paper Vote Privacy Defence". The main project can be found here: https://github.com/anettehabanen/Deep_Fakes_for_Paper_Vote_Privacy_Defence

### Clone the repository:
```
git clone https://github.com/anettehabanen/Change_My_Vote
cd Change_My_Vote
```

### Create Python environment

NB! This works if you have Python version 3.12 or lower. To check the Python version:
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
reflex run
```

The webpage can be found at http://localhost:3001/

If the port 8001 or 3001 is already in use, go to the [config file](rxconfig.py) and change them to open ports.

## Example videos
The folder 'example_videos' has some images and videos of ballots and digits. To test the Reflex app, use these files when asked to upload the image of 3-digit code and a video of a voting ballot. Testing with your own videos/images is also possible.
