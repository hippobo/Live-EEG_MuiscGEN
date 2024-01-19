Flask app to generate music using 2 models: concatenated-transformer based on [continous-concatenated](https://github.com/serkansulun/midi-emotion) transformer model, using the 4 quadrant emotion system trained on the EMOPIA dataset and REMIPlus encoded MIID files, and a no conditioning decoder-only transformer trained on the lahk_midi clean dataset.  

Screenshot of app using Arousal/Valence Quadrant values to condition generation

![App screenshot using quadrants](docs/flask_screen0.png)


## TODO:

- [x] 3D visualization using three.js
- [x] Music generation using quadrants
- [x] Flask integration
- [] Connection + Datastream w/Muse
- [] Record datastreal w/Muse (.csv) using muselsl
- [] Preprocess incoming Muse Data (using FACED dataset) --> to .pkl then DE + norm + LDS. 
- [] Use preprocessed data (pkl.) for visualization --> send to frontend
- [] Inference using trained SVM w/ FACED data --> send to music model (backend) + display frontend

