let model;
let isModelLoaded = false;

const l2Regularizer = (lambda) => tf.regularizers.l2(lambda);

// Nombres de las clases
const classNames = ['Benigno', 'Maligno'];

// Cargar el modelo
async function loadModel() {
    try {
        updateStatus('Cargando Modelo...', 'loading');
        
        // Cargar el modelo real (no dummy)
        model = await tf.loadGraphModel('model.json');

        // Simular breve carga visual
        await new Promise(resolve => setTimeout(resolve, 1500));

        isModelLoaded = true;
        updateStatus('✅ Modelo Listo', 'ready');
        enableButtons();
        console.log("✅ ¡Modelo cargado exitosamente!");
    } catch (error) {
        console.error("Error al cargar el modelo:", error);
        updateStatus('❌ Error al Cargar Modelo', 'error');
        document.getElementById('result-label').textContent =
            'Error al cargar el modelo. Por favor, recarga la página.';
    }
}

// Estado visual del modelo
function updateStatus(message, type) {
    const statusEl = document.getElementById('modelStatus');
    statusEl.className = `status-indicator status-${type}`;
    statusEl.innerHTML = type === 'loading'
        ? `<span class="loading"></span> ${message}`
        : message;

    if (type === 'ready') {
        setTimeout(() => {
            statusEl.style.opacity = '0';
            setTimeout(() => (statusEl.style.display = 'none'), 300);
        }, 2000);
    }
}

// Habilitar botón de subida
function enableButtons() {
    document.getElementById('predict-button').disabled = false;
}

// Subir imagen
document.getElementById('image-upload').addEventListener('change', handleImageUpload);

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = function () {
        displayImage(img);
        processImage(img);
    };
    img.src = URL.createObjectURL(file);
}

// Mostrar vista previa
function displayImage(img) {
    const previewImg = document.getElementById('input-image');
    const placeholder = document.getElementById('placeholderText');
    const container = document.getElementById('imageContainer');

    placeholder.classList.add('hidden');
    previewImg.src = img.src;
    previewImg.classList.remove('hidden');
    container.classList.add('has-image');
}

// Procesar imagen y predecir
async function processImage(image) {
    if (!isModelLoaded) {
        document.getElementById('result-label').textContent =
            'El modelo aún se está cargando. Por favor, espera...';
        return;
    }

    try {
        // Mostrar estado de análisis
        document.getElementById('result-label').innerHTML =
            '<span class="loading"></span> Analizando imagen...';

        const imgTensor = tf.tidy(() => {
            let tensor = tf.browser.fromPixels(image)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(tf.scalar(255));
    
            // Detecta si el modelo fue exportado en formato canales-primeros
            if (model.inputs && model.inputs[0].shape.length === 4 && model.inputs[0].shape[1] === 3) {
                // El modelo espera [1,3,224,224] → permutamos ejes
                tensor = tensor.transpose([2, 0, 1]);
            }
        
            return tensor.expandDims(0); // Devuelve [1,3,224,224] o [1,224,224,3] según corresponda
        });


        // Predicción compatible (GraphModel o LayersModel)
        const prediction = model.executeAsync
            ? await model.executeAsync(imgTensor)
            : await model.predict(imgTensor);

        const predictionData = await prediction.data();
        tf.dispose([imgTensor, prediction]);

        // Determinar clase más probable
        const maxIdx = predictionData.indexOf(Math.max(...predictionData));
        const confidence = predictionData[maxIdx];
        const predictedClass = classNames[maxIdx];

        displayPrediction(predictedClass, confidence);
    } catch (error) {
        console.error("Error al hacer la predicción:", error);
        document.getElementById('result-label').textContent =
            'Error al analizar la imagen. Por favor, intenta de nuevo.';
    }
}

// Mostrar resultado
function displayPrediction(prediction, confidence) {
    const resultEl = document.getElementById('result-label');
    const confidenceBar = document.getElementById('result-confidence');
    const confidenceFill = document.getElementById('confidenceFill');

    const isMalignant = prediction === 'Maligno';
    const icon = isMalignant ? '⚠️' : '✅';
    const color = isMalignant
        ? 'linear-gradient(45deg, #ff6b6b, #ee5a52)'
        : 'linear-gradient(45deg, #a3d977, #68d391)';

    resultEl.className = `prediction-result result-${prediction.toLowerCase()}`;
    resultEl.innerHTML = `
        <div style="font-size: 1.5rem; margin-bottom: 10px;">
            ${icon} ${prediction}
        </div>
        <div style="font-size: 1rem; opacity: 0.8;">
            Confianza: ${(confidence * 100).toFixed(1)}%
        </div>
    `;

    confidenceBar.classList.remove('hidden');
    confidenceFill.style.width = `${confidence * 100}%`;
    confidenceFill.style.background = color;
}

// Inicializar
window.onload = loadModel;
console.log("Forma de entrada del modelo:", model.inputs[0].shape);


