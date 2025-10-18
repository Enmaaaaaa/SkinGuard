// =======================================================
// 🧠 SKINGUARD - SCRIPT.JS (versión corregida sin reescribir todo)
// =======================================================

// 🔹 Asegúrate de tener tf.js cargado en index.html:
// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0"></script>

let model;

// =========================
// 1️⃣ CARGA DEL MODELO
// =========================
async function loadModel() {
  try {
    model = await tf.loadGraphModel('model.json');
    console.log("✅ Modelo cargado correctamente");
  } catch (error) {
    console.error("❌ Error al cargar el modelo:", error);
  }
}

// =========================
// 2️⃣ PREPROCESAR IMAGEN
// =========================
function preprocessImage(imageElement) {
  try {
    const tensor = tf.browser.fromPixels(imageElement)
      .resizeNearestNeighbor([224, 224])  // Redimensionar a 224x224
      .toFloat()
      .div(tf.scalar(255))                // Normalizar a rango 0–1
      .expandDims();                      // [1, 224, 224, 3]
    return tensor;
  } catch (err) {
    console.error("Error en preprocessImage:", err);
  }
}

// =========================
// 3️⃣ HACER PREDICCIÓN
// =========================
async function predict() {
  try {
    const img = document.getElementById("input-image");
    if (!model) {
      alert("Primero carga el modelo antes de predecir.");
      return;
    }

    // 🔥 CORREGIDO: preprocesamiento adecuado
    const tensor = preprocessImage(img);

    // 🔥 CORREGIDO: uso de .predict() y .data() correcto
    const prediction = await model.predict(tensor);
    const result = await prediction.data();

    // Tus etiquetas originales (ajusta según tu dataset)
    const labels = ["Benigno", "Maligno"];
    const maxIndex = result.indexOf(Math.max(...result));

    // Mostrar resultados
    const label = labels[maxIndex];
    const confidence = (result[maxIndex] * 100).toFixed(2);

    document.getElementById("result-label").innerText = `🧬 Resultado: ${label}`;
    document.getElementById("result-confidence").innerText = `Confianza: ${confidence}%`;

    console.log(`✅ Predicción: ${label} (${confidence}%)`);
  } catch (error) {
    console.error("Error al hacer la predicción:", error);
  }
}

// =========================
// 4️⃣ CARGAR IMAGEN LOCAL
// =========================
document.getElementById("image-upload").addEventListener("change", function (event) {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      const img = document.getElementById("input-image");
      img.src = e.target.result;
      document.getElementById("result-label").innerText = "";
      document.getElementById("result-confidence").innerText = "";
    };
    reader.readAsDataURL(file);
  }
});

// =========================
// 5️⃣ EVENTO BOTÓN
// =========================
document.getElementById("predict-button").addEventListener("click", predict);

// =========================
// 6️⃣ INICIALIZACIÓN
// =========================
loadModel();
