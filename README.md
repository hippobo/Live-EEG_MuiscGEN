Flask app to generate music using 2 models: concatenated-transformer based on [continous-concatenated](https://github.com/serkansulun/midi-emotion) transformer model, using the 4 quadrant emotion system trained on the EMOPIA dataset and REMIPlus encoded MIID files, and a no conditioning decoder-only transformer trained on the lahk_midi clean dataset.  

A live 3D visualization coded using three.js allows the user to see live brain activity. 

The circle in the 4 quadrants changes position and size based on the inferred user's emotional state. The position indicates the user's valence and arousal values and the circle size on the dominance value (intensity).

Music is generated based on these values, using a model trained on the DEAP dataset using the Muse headset channels.


Image of app using Arousal/Valence Quadrant values to condition generation

![Gif_live](docs/gif_flask_app_live.gif)



# Installation

- Clone this repo

- Create conda env with python :
``` python
conda create --name myenv -c conda-forge python=3.11.4
conda activate myenv
```
- Install requirements :

```python
pip install -r requirements
```
- Run flask 

```python
cd flask_app
flask run
```

# To use 

- ## Demo Tab
    The left tab is used to showcase the 32 channel example 3D visualization and music generation possibilities. Click on the checkbox to condition the music generation on emotions (from 1 to 4), and choose sequence length and click "Generate Sequence". You can then add to the generated sequence by choosing a length and clicking "Add to sequence". 

    Leave the checkbox unchecked to generate music using the lahk_midi clean dataset without conditioning. You can use the slider to change the temperature value of the music generation model.

    You can download your generated MIDI sequence by clicking on the "Download Sequence" button. 

    The 3D visualization has 10 30-second examples pre-loaded of live EEG data from the FACED dataset. 

- ## Live EEG Tab

     The right tab is used for live streaming EEG data from the muse 2 EEG headset. Activate bluetooth on your device and click on "Connect EEG" to connect, until "Processing EEG..." appears. You will see the 4 muse channels light up in the 3D visualization and music generated and played automatically in your browser. If no music is heard but notes are seen in the MIDI visualizer, click on the webpage to allow the audio context to work.

    





