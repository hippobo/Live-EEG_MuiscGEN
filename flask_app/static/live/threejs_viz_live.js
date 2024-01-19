import * as THREE from 'three';

import Stats from 'three/addons/libs/stats.module.js';

import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SubsurfaceScatteringShader } from 'three/addons/shaders/SubsurfaceScatteringShader.js';
import {OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';



let container, camera, scene, renderer;
let model, model2;
let brainGroup = new THREE.Group();
let lights = {};
let textMesh;
let globalEEGData = null;
let currentDataPoint = 0;
let maxDataPoints = 0;
let lowerBound, upperBound;
let interpolationFrames = 120;


const updateChannels = [20, 0, 1, 21];



const lightNames = ['fp1', 'fp2', 'fz', 'f3', 'f4', 'f7', 'f8', 'fc1', 'fc2', 'fc5', 'fc6', 'cz', 'c3', 'c4', 't3', 't4', 'a1', 'a2', 'cp1', 'cp2', 'cp5', 'cp6', 'pz', 'p3', 'p4', 't5', 't6', 'po3', 'po4', 'oz', 'o1', 'o2'];


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

init();
animate();

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
	uniforms[ 'thicknessPower' ].value = 0.01;
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

function init() {
    container = document.getElementById('ThreeJsCanvas');
    camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 1, 5000);
    camera.position.set(0, 50, 2000);

    scene = new THREE.Scene();
    scene.add(brainGroup);

    const ambientLight = new THREE.AmbientLight(0xc1c1c1, 3);
    scene.add(ambientLight);

    lightNames.forEach(name => {
        lights[name] = createLight(name);
    });

    periodicallyFetchEEGData(); 

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setClearColor(0x121212);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 500;
    controls.maxDistance = 3000;

    window.addEventListener('resize', onWindowResize);

    loadFont();
	initMaterial();
}

function createLight(name) {
    let lightMesh = new THREE.Mesh(new THREE.SphereGeometry(0.001, 8, 8), new THREE.MeshBasicMaterial({ color: 0xc1c100 }));
    const pointLight = new THREE.PointLight(0xc1c100, 0.5, 50, 0);
    lightMesh.add(pointLight);
    brainGroup.add(lightMesh);
    return lightMesh;
}

function loadFont() {
    const fontLoader = new FontLoader(manager);
    fontLoader.load('https://threejs.org/examples/fonts/droid/droid_sans_regular.typeface.json', function(font) {
        updateText("Brain Activity Live", font);
    });
}

function updateText(newText, font) {
    if (textMesh) {
        scene.remove(textMesh);
    }
    const textGeometry = new TextGeometry(newText, {
        font: font,
        size: 40,
        height: 10,
        curveSegments: 12,
        bevelEnabled: false
    });
    const textMaterial = new THREE.MeshBasicMaterial({ color: 0xc1c100 });
    textMesh = new THREE.Mesh(textGeometry, textMaterial);
    textMesh.position.set(-250, 200, 0);
    scene.add(textMesh);
}
function fetchEEGData() {
    fetch('./eeg_data_live')
        .then(response => response.json())
        .then(eegData => {
            globalEEGData = eegData.channel_data;
            maxDataPoints = globalEEGData[0].length;
            const percentiles = findPercentiles(globalEEGData, 5, 95);
            lowerBound = percentiles.lowerBound;
            upperBound = percentiles.upperBound;
            currentDataPoint = 0;   
        })
        .catch(error => {
            console.error("Error fetching EEG data:", error);
        });
}


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




function findPercentiles(data, lowerPercentile, upperPercentile) {
    // Flatten the data across all channels
    const flattenedData = data.reduce((acc, val) => acc.concat(val), []);

    // Sort the flattened data
    const sortedData = flattenedData.sort((a, b) => a - b);

    // Find the indices for the lower and upper percentiles
    const lowerIndex = Math.floor(lowerPercentile / 100.0 * (sortedData.length - 1));
    const upperIndex = Math.floor(upperPercentile / 100.0 * (sortedData.length - 1));

    return { 
        lowerBound: sortedData[lowerIndex], 
        upperBound: sortedData[upperIndex] 
    };
}


function normalizeValue(value, lowerBound, upperBound) {
    return (10 - (-10)) * ((value - lowerBound) / (upperBound - lowerBound)) - 10;
}

function updateLightIntensities() {
    if (!globalEEGData || currentDataPoint >= maxDataPoints) return;

    updateChannels.forEach(lightIndex => {
        const lightName = lightNames[lightIndex];
        if (lights[lightName] && lights[lightName].children[0]) {
            // Assuming each light corresponds to a channel in a specific order
            const channelIndex = lightIndex % globalEEGData.length; // Adjust this logic based on your data structure
            const rawIntensity = globalEEGData[channelIndex][currentDataPoint];

            lights[lightName].targetIntensity = normalizeValue(rawIntensity, lowerBound, upperBound);

            // Frame interpolation
            lights[lightName].children[0].intensity += (lights[lightName].targetIntensity - lights[lightName].children[0].intensity) / interpolationFrames;
        }
    });

    currentDataPoint++;
}




function periodicallyFetchEEGData() {
    fetchEEGData();
    console.log("Fetching EEG data...");    
    setTimeout(periodicallyFetchEEGData, 5000); // Fetch data every 5 seconds
}

function initGUI( uniforms ) {

	const gui = new GUI( { title: 'Thickness Control' } );


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
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    render();
}

function render() {
    updateLightIntensities();
    brainGroup.rotation.y += 0.0005;
    renderer.render(scene, camera);
}
