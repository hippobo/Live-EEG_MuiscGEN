


let temperatureValue = 0.7;
const maxValue = 200;



let sequenceGenerated = false;
let globalContext = [221]; //initial context



const midiPlayer = document.querySelector('midi-player');
const midiVisualizer = document.querySelector('midi-visualizer');



const quadrantsDiv = document.getElementById('quadrants');
quadrantsDiv.style.display = 'block';

document.addEventListener('DOMContentLoaded', function() {
    let statusInterval = null;

    function updateEEGStatus() {
        fetch('/eeg_status', { method: 'GET' })
            .then(response => response.json())
            .then(data => {
                document.getElementById('successText').innerText = data.message;
                // Additional UI updates based on data.status can be added here
            })
            .catch(error => {
                console.error('Error fetching EEG status:', error);
                clearInterval(statusInterval); // Stop polling if there's an error
            });
    }

    document.getElementById('connectEEGButton').addEventListener('click', function() {
        // Show spinner
        document.getElementById('spinner').style.display = 'inline-block';

        // Make a request to the Flask endpoint
        fetch('/connect_eeg', { method: 'GET' })
            .then(response => response.json())
            .then(data => {
                // Hide spinner
                document.getElementById('spinner').style.display = 'none';

                if(data.status === 'success') {
                    // Show success message
                    document.getElementById('successText').innerText = data.message;
                    document.getElementById('successText').classList.remove('text-danger');
                    document.getElementById('successText').classList.add('text-success');
                    document.getElementById('successText').classList.remove('visually-hidden');

                    // Change the button to a success state
                    document.getElementById('connectEEGButton').classList.remove('btn-primary');
                    document.getElementById('connectEEGButton').classList.add('btn-success');

                    // Start polling for EEG status updates every 2 seconds
                    statusInterval = setInterval(updateEEGStatus, 5000);
                } else if(data.status === 'error') {
                    // Show error message
                    document.getElementById('successText').innerText = 'Failed to connect to EEG headset: ' + data.message;
                    document.getElementById('successText').classList.add('text-danger');
                    document.getElementById('successText').classList.remove('visually-hidden');
                }
            })
            .catch(error => {
                console.error('Fetch operation error:', error);
                document.getElementById('spinner').style.display = 'none';
                document.getElementById('successText').innerText = 'Network error: ' + error;
                document.getElementById('successText').classList.add('text-danger');
                document.getElementById('successText').classList.remove('visually-hidden');
            });
    });
});


let quadrantCounts = [0,0,0,0];

 // Get the canvas and context
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');


        // Function to draw the initial quadrants
        function drawQuadrants() {
            const width = canvas.width;
            const height = canvas.height;

            
            ctx.strokeStyle = 'white'

            // Draw horizontal line
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);

            // Draw vertical line
            ctx.moveTo(width / 2, 0);
            ctx.lineTo(width / 2, height);

            // Stroke the lines
            ctx.lineWidth = 5;
            ctx.stroke();

            ctx.font = '20px Arial';
            ctx.fillStyle = 'white';

            ctx.fillText("Angry - LVHA",  width * 0.025, height * 0.05); // Top-left
            ctx.fillText("Happy - HVHA", width * 0.52, height * 0.05); // Top-right
            ctx.fillText("Sad - LVLA", width * 0.025, height * 0.55); // Bottom-left
            ctx.fillText("Relaxed - HVLA", width * 0.52  , height * 0.55); // Bottom-right
        }

        // Function to update the tally for a quadrant
        function updateTally(x, y, dominance) {
            const width = canvas.width;
            const height = canvas.height;

            // Determine the quadrant
            let quadrant;
            if (x < width / 2 && y < height / 2) {
                quadrant = 1; // Top-left
            } else if (x >= width / 2 && y < height / 2) {
                quadrant = 0; // Top-right
            } else if (x < width / 2 && y >= height / 2) {
                quadrant = 2; // Bottom-left
            } else {
                quadrant = 3; // Bottom-right
            }
         
            if (dominance > 1) {
            quadrantCounts[quadrant] = Math.round(dominance - 1) * (4 / (9 - 1));
            }

            // Redraw the quadrant tallies
            drawTallies();
        }

        
        function drawTallies() {
            const width = canvas.width;
            const height = canvas.height;

            // Clear the canvas first
            ctx.clearRect(0, 0, width, height);

            // Redraw the quadrants
            drawQuadrants();

            // Set font for tally text
            ctx.font = '20px Arial';
            ctx.fillStyle = 'white';

            // Draw the tally for each quadrant
            ctx.fillText(quadrantCounts[1], width * 0.25, height * 0.25); // Top-left
            ctx.fillText(quadrantCounts[0], width * 0.75, height * 0.25); // Top-right
            ctx.fillText(quadrantCounts[2], width * 0.25, height * 0.75); // Bottom-left
            ctx.fillText(quadrantCounts[3], width * 0.75, height * 0.75); // Bottom-right
        }

    
               

        // Initial draw
        drawTallies();



const downloadButton = document.getElementById('downloadButton');
let midiBlob; 




function fetchAndDisplayEmotions() {
    fetch('/eeg_emotions')
        .then(response => response.json())
        .then(data => {
            const { valence, arousal, dominance } = data;
            drawPointOnCanvas(valence, arousal, dominance);
        })
        .catch(error => console.error('Error fetching EEG emotions:', error));
}

function drawPointOnCanvas(valence, arousal, dominance) {
    const width = canvas.width;
    const height = canvas.height;
  
    // Normalize valence and arousal to canvas coordinates
    const x = (valence - 1) / 8 * width;
    const y = (9 - arousal) / 8 * height;  // Invert arousal to match canvas coordinates

    // Clear previous point
    ctx.clearRect(0, 0, width, height);



    // Redraw the quadrants
    drawQuadrants();

    // Draw the point
    ctx.beginPath();
    ctx.arc(x, y, dominance * 10, 0, 2 * Math.PI); // Dominance affects the size
    ctx.fillStyle = 'red';
    ctx.fill();

    updateTally(x,y, dominance);

    // Redraw tallies
    
}

setInterval(fetchAndDisplayEmotions, 2000);
setInterval(generateAndPlayMidi, 5000);

function updateMidiSource() {
    var input = document.getElementById('midiFileInput');
    var file = input.files[0];
    if (file) {
        var reader = new FileReader();

        reader.onload = function(e) {
         

            midiPlayer.src = e.target.result;
            midiVisualizer.src = e.target.result;
            
            // midiPlayer.stop();
            // midiPlayer.start();
        };

        reader.readAsDataURL(file);
    }
}



document.getElementById('midiFileInput').addEventListener('change', updateMidiSource);





function generateAndPlayMidi() {
        const requestPayload = {
            sequence_length: maxValue,
            context: globalContext,
            quadrant_counts: quadrantCounts,
            temperatureValue: temperatureValue
        };
    
        fetch('/generate_midi_live', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestPayload)
        })
        .then(response => response.json())
        .then(data => {
            const newSequence = data.context;
                globalContext = newSequence;
                console.log(globalContext);

               
                downloadButton.style.display = 'block';
                
                downloadButton.href = data.midi_file_url;
                downloadButton.download = 'generated_midi_seq.mid';
                
                midiPlayer.src = data.midi_file_url;
                midiVisualizer.src = data.midi_file_url;

                midiPlayer.stop();
                midiPlayer.start();
        })
        .catch(error => {
            console.error('Error generating MIDI:', error);
        });
    }
    


