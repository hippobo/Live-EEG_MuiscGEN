import * as THREE from 'three';

import Stats from 'three/addons/libs/stats.module.js';

import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SubsurfaceScatteringShader } from 'three/addons/shaders/SubsurfaceScatteringShader.js';
import {OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';



		let container, stats;
		let camera, scene, renderer;
		let model;
        let model2;
		const lights = {};
		let lowerBound;
		let upperBound;
		let textMesh;
		let brainGroup = new THREE.Group();
		let lightMesh;


		let globalEEGData = null;
		let currentDataPoint = 0; 
		let videoClipIndex = 0;
	let interpolationFrames = 30;
		let maxDataPoints = 0;



	const canvas = document.getElementById('ThreeJsCanvas');
	const lightNames = ['fp1', 'fp2', 'fz', 'f3', 'f4', 'f7', 'f8', 'fc1', 'fc2', 'fc5', 'fc6' , 'cz', 'c3', 'c4', 't3', 't4', 'a1' ,'a2', 'cp1', 'cp2', 'cp5', 'cp6', 'pz', 'p3','p4', 't5', 't6', 'po3', 'po4', 'oz', 'o1', 'o2'];


const manager = new THREE.LoadingManager();
const loadingScreen = document.createElement('div');
loadingScreen.classList.add('loading-screen');
loadingScreen.innerHTML = '<div class="loading-content"><p>Loading...</p><div class="spinner"></div></div>';
document.body.appendChild(loadingScreen);

manager.onStart = function (url, itemsLoaded, itemsTotal) {
    loadingScreen.style.display = 'flex';
};

manager.onLoad = function () {
    loadingScreen.style.display = 'none';
};

manager.onProgress = function (url, itemsLoaded, itemsTotal) {
};

manager.onError = function (url) {
    console.error('There was an error loading ' + url);
};
const fontLoader = new FontLoader(manager);
	

function createLight(name) {
	lightMesh = new THREE.Mesh(new THREE.SphereGeometry(0.001, 8, 8), new THREE.MeshBasicMaterial({ color: 0xc1c100 }));
	const pointLight = new THREE.PointLight(0xc1c100, 0.5, 50, 0);
	lightMesh.add(pointLight); 
	brainGroup.add(lightMesh)
	return lightMesh;
}

	

	

function updateText(newText, font) {
	if (textMesh) {
		scene.remove(textMesh); // Remove the existing text mesh
	}
	const textGeometry = new TextGeometry(newText, {
		font: font,
		size: 40,
		height: 10,
		curveSegments: 12,
		bevelEnabled: false,
		bevelThickness: 3,
		bevelSize: 5,
		bevelOffset: 1,
		bevelSegments: 10
	});
	const textMaterial = new THREE.MeshBasicMaterial({ color: 0xc1c100 });
	textMesh = new THREE.Mesh(textGeometry, textMaterial);
	textMesh.position.set(-250, 200, 0); // Adjust position as needed
	scene.add(textMesh);
}

		
		
init();

fontLoader.load('https://threejs.org/examples/fonts/droid/droid_sans_regular.typeface.json', function ( font ) {
	
updateText( "Brain Activity Live", font);
});

animate();



		
	

function init() {

	container = document.getElementById('ThreeJsCanvas')

	camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 1, 5000);
	camera.position.set( 0, 50, 2000 );
	

	scene = new THREE.Scene();
	scene.add(brainGroup)

	// Lights
	const ambient_light = new THREE.AmbientLight( 0xc1c1c1 , 3)
	scene.add( ambient_light );

	

	
lightNames.forEach(name => {
	lights[name] = createLight(name);
});





// Fetch the EEG data
fetch('./eeg_data_live')
	.then((response) => response.json())
	.then((eegData) => {
		globalEEGData = eegData; // Store the data in the global variable
			// Flatten the EEG data to find percentiles
		const flatEEGData = flattenEEGData(globalEEGData);
		const percentiles  = findPercentiles(flatEEGData, 5, 95); // Using 5th and 95th percentiles
		lowerBound = percentiles.lowerBound;
		upperBound = percentiles.upperBound;
		maxDataPoints = globalEEGData[videoClipIndex][0].length;
		

	})
	.catch((error) => {
		console.error("Error fetching EEG data:", error);
	});




			

		lights['fp1'].position.set(-33, -16, -102);
		lights['fp2'].position.set(33, -16, -102);

		lights['f7'].position.set(-60, -16, -60);
		lights['f8'].position.set(60, -16, -60);


		lights['f3'].position.set(-40, 30, -60);
		lights['f4'].position.set(40, 30, -60);

		lights['fz'].position.set(0, 50, -60);
		lights['pz'].position.set(0, 60, 70);

		lights['a1'].position.set(-90, -16, 0);
		lights['a2'].position.set(90, -16, 0);


		lights['t3'].position.set(-66, 0, 0);
		lights['t4'].position.set(66, 0, 0);


		lights['c3'].position.set(-60, 50, 0);
		lights['cz'].position.set(0, 80, 0);
		lights['c4'].position.set(60, 40, 0);


		lights['t5'].position.set(-80,-10, 30);
		lights['p3'].position.set(-60, 50, 70);

		lights['p4'].position.set(60, 50, 70);
		lights['t6'].position.set(80,-10, 30);
		lights['o1'].position.set(-40, -10, 90);
		lights['o2'].position.set(40, -10, 90);
		lights['oz'].position.set(0, -5, 100);
		lights['fc1'].position.set(-25, 50, -30);
		lights['fc2'].position.set(25, 50, -30);
		lights['fc5'].position.set(-60, 20, -20);
		lights['fc6'].position.set(60, 20, -20);
		lights['cp1'].position.set(-25, 60, 40);
		lights['cp2'].position.set(25, 60, 40);
		lights['cp5'].position.set(-50, 40, 50);
		lights['cp6'].position.set(50, 40, 50);

		lights['po3'].position.set(-30, 15, 80);
		lights['po4'].position.set(30, 15, 80);




		renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
		renderer.setClearColor(0x121212); 
		renderer.setSize(container.clientWidth, container.clientHeight);
		container.appendChild(renderer.domElement);
		container.appendChild( renderer.domElement );

			//


	// Helpers
const axesHelper = new THREE.AxesHelper( 1000);
axesHelper.setColors(0xff0000, 0x00ff00, 0x0000ff); // R, G, B
// scene.add(axesHelper);

	// stats = new Stats();
	// container.appendChild( stats.dom );

	const controls = new OrbitControls( camera,renderer.domElement );
	controls.enableDamping =true;
	controls.minDistance = 500;
	controls.maxDistance = 3000;

	initMaterial();

	

	onWindowResize();	

}

		function initMaterial() {

			const loader = new THREE.TextureLoader(manager);
			const imgTexture = loader.load( 'static/models/white.jpg' );
			imgTexture.colorSpace = THREE.SRGBColorSpace;

			const thicknessTexture = loader.load( 'static/models/white.jpg' );
			imgTexture.wrapS = imgTexture.wrapT = THREE.RepeatWrapping;

			const shader = SubsurfaceScatteringShader;
			const uniforms = THREE.UniformsUtils.clone( shader.uniforms );

			uniforms[ 'map' ].value = imgTexture;

            //default values
			uniforms[ 'diffuse' ].value = new THREE.Vector3( 1.0, 0.2, 0.2 );
			uniforms[ 'shininess' ].value = 500;
            
            // default color values
			uniforms[ 'thicknessMap' ].value = thicknessTexture;
			uniforms[ 'thicknessColor' ].value = new THREE.Vector3( 0.5, 0.3, 0.0 );


			uniforms[ 'thicknessDistortion' ].value = 0.25;
			uniforms[ 'thicknessAmbient' ].value = 0.0;
			uniforms[ 'thicknessAttenuation' ].value = 0.1;
			uniforms[ 'thicknessPower' ].value = 2.0;
			uniforms[ 'thicknessScale' ].value = 25.0;

			const material = new THREE.ShaderMaterial( {
				uniforms: uniforms,
				vertexShader: shader.vertexShader,
				fragmentShader: shader.fragmentShader,
				lights: true
			} );
			material.extensions.derivatives = true;

			// LOADER

            const loaderOBJ = new OBJLoader(manager);
            loaderOBJ.load('static/models/brain_model.obj', function ( object ) {
                model = object.children[ 0 ];
                model.position.set( 0, 0, 0 );
                model.scale.setScalar( 50);
                model.material = material;

               model2 = model.clone();
                model2.scale.x *= -1;


				brainGroup.add(model);
				brainGroup.add(model2);
               

            } );
			initGUI( uniforms );

		}


		function flattenEEGData(eegData) {
			return eegData.flat(2); // Flatten the 3D array into a 1D array
		}
		
		function findPercentiles(data, lowerPercentile, upperPercentile) {
			const sortedData = [...data].sort((a, b) => a - b);
			const lowerIndex = Math.floor(lowerPercentile / 100.0 * (sortedData.length - 1));
			const upperIndex = Math.floor(upperPercentile / 100.0 * (sortedData.length - 1));
			
			return {'lowerBound' : sortedData[lowerIndex],
			'upperBound' : sortedData[upperIndex] };
		}
		
		function normalizeValue(value, lowerBound, upperBound) {	
			
			return  (10 - (-10)) * ((value - lowerBound)/(upperBound - lowerBound)) - 10;
		}
		
		document.getElementById('randomButton').addEventListener('click', () => {
			const maxVideoClips = 10; 
			videoClipIndex = Math.floor(Math.random() * maxVideoClips);
			currentDataPoint = 0; 
			document.getElementById('exampleNum').innerText = videoClipIndex; 
		});
		
		document.getElementById('interpolationSlider').addEventListener('input', (event) => {
			const sliderValue = event.target.value;
			interpolationFrames = sliderValue; // Update interpolation frames based on slider
		});

	
	
	
		
		
	
	lightNames.forEach(name => {
		lights[name].currentIntensity = lights[name].children[0].intensity; // Store the initial intensity
		lights[name].targetIntensity = lights[name].children[0].intensity; // Set a target intensity
	});
	
	function updateLightIntensities() {
		if (!globalEEGData || !globalEEGData.length) return;
	
		const numberOfElectrodes = 32;
		if (currentDataPoint < globalEEGData[videoClipIndex][0].length) {
			for (let i = 0; i < numberOfElectrodes; i++) {
				if (lights[lightNames[i]] && lights[lightNames[i]].children[0]) {
					const electrodeData = globalEEGData[videoClipIndex][i];
					const rawIntensity = electrodeData[currentDataPoint];
					lights[lightNames[i]].targetIntensity = normalizeValue(rawIntensity, lowerBound, upperBound);
	
					// Perform linear interpolation
					lights[lightNames[i]].currentIntensity += (lights[lightNames[i]].targetIntensity - lights[lightNames[i]].currentIntensity) / interpolationFrames;
					lights[lightNames[i]].children[0].intensity = lights[lightNames[i]].currentIntensity;
				}
			}
			currentDataPoint++;
		}
	}
		function initGUI( uniforms ) {

			const gui = new GUI( { title: 'Thickness Control' } );

			// const gui2 = new GUI({ title: 'Light Controls' });

			// // Iterate over each light to create a folder and add controls
			// Object.keys(lights).forEach(lightName => {
			// 	const lightFolder = gui.addFolder(lightName);
			// 	const pointLight = lights[lightName].children[0];
		
			// 	// Add controls for intensity and distance
			// 	lightFolder.add(pointLight, 'intensity', 0, 10).name('Intensity');
			// 	lightFolder.add(pointLight, 'distance', 0, 1000).name('Distance');
			// 			});
        


			const ThicknessControls = function () {

				this.distortion = uniforms[ 'thicknessDistortion' ].value;
				this.ambient = uniforms[ 'thicknessAmbient' ].value;
				this.attenuation = uniforms[ 'thicknessAttenuation' ].value;
				this.power = uniforms[ 'thicknessPower' ].value;
				this.scale = uniforms[ 'thicknessScale' ].value;

			};

			const thicknessControls = new ThicknessControls();

			gui.add( thicknessControls, 'distortion' ).min( 0.01 ).max( 1 ).step( 0.01 ).onChange( function () {

				uniforms[ 'thicknessDistortion' ].value = thicknessControls.distortion;

			} );

			gui.add( thicknessControls, 'ambient' ).min( 0.01 ).max( 5.0 ).step( 0.05 ).onChange( function () {

				uniforms[ 'thicknessAmbient' ].value = thicknessControls.ambient;

			} );

			gui.add( thicknessControls, 'attenuation' ).min( 0.01 ).max( 5.0 ).step( 0.05 ).onChange( function () {

				uniforms[ 'thicknessAttenuation' ].value = thicknessControls.attenuation;

			} );

			gui.add( thicknessControls, 'power' ).min( 0.01 ).max( 16.0 ).step( 0.1 ).onChange( function () {

				uniforms[ 'thicknessPower' ].value = thicknessControls.power;

			} );

			gui.add( thicknessControls, 'scale' ).min( 0.01 ).max( 50.0 ).step( 0.1 ).onChange( function () {

				uniforms[ 'thicknessScale' ].value = thicknessControls.scale;

			} );

		}

		function onWindowResize() {
			const width = container.clientWidth;
			const height = container.clientHeight;
		
			// Update camera aspect ratio and renderer size
			camera.aspect = width / height;
			camera.updateProjectionMatrix();
			renderer.setSize(width, height);
		}
		
		window.addEventListener('resize', onWindowResize);

		function animate() {

			requestAnimationFrame( animate );

			render();

			// stats.update();

		}

		function render() {

			
			updateLightIntensities();

			brainGroup.rotation.y += 0.0005;


			renderer.render( scene, camera );

		}