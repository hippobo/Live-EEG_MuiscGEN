


let temperatureValue = 0.7;



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

            // Increment the count for the quadrant and cycle back to 0 if it's currently 4
            quadrantCounts[quadrant] = Math.round(dominance - 1) * (4 / (9 - 1));

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

        // Event listener for canvas click
                canvas.addEventListener('click', function(event) {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            updateTally(x, y);
        });

        // Initial draw
        drawTallies();



const downloadButton = document.getElementById('downloadButton');
let midiBlob; 


    function updateButtonText() {
    
    generateButton.textContent = sequenceGenerated ? 'Add to Sequence' : 'Generate MIDI Sequence';
        
}


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
const tokenDisplay = document.getElementById("text");




// document.getElementById('generateButton').addEventListener('click', async () => {
//         await Tone.start()
//         console.log('audio is ready')
        
//         const maxValue = parseInt(document.getElementById('maxValueInput').value);
        
//         if (isNaN(maxValue) || maxValue <= 0) {
//             alert('Please enter a valid sequence length.');
//             return;
//         }
        
//         const requestPayload = {
//             sequence_length: maxValue,
//             context: globalContext,
//             quadrant_use : useVA,
//             quadrant_counts : quadrantCounts,
//             temperatureValue : temperatureValue
//         };


//         try {
//             const response = await fetch('/generate_midi_demo', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json'
//                 },
//                 body: JSON.stringify(requestPayload)
//             });

//             if (response.ok) {
//                 const responseData = await response.json();

                
//                 const newSequence = responseData.context;
//                 globalContext = newSequence;

//                 // Update the download button with the link to the MIDI file
//                 downloadButton.style.display = 'block';
                
//                 downloadButton.href = responseData.midi_file_url;
//                 downloadButton.download = 'generated_midi_seq.mid';
                
//                 midiPlayer.src = responseData.midi_file_url;
//                 midiVisualizer.src = responseData.midi_file_url;
//                 console.log(midiPlayer);

//                 midiPlayer.stop();
//                 midiPlayer.start();
//                 // Update generate button text
//                 sequenceGenerated = true;
//                 updateButtonText();
//             } else {
//                 throw new Error('Failed to generate MIDI file');
//             }
//         } catch (error) {
//             console.error(error.message);
//         }
//     });




