// Real three-class TensorFlow.js classifier.
// Output order is fixed: 0 ORGANIC, 1 INORGANIC, 2 REUSABLE.

const WASTE_CATEGORIES = {
    ORGANIC: {
        id: "ORGANIC", name: "Organic", bin: "ORGANIC BIN",
        physicalBin: "organic", recommendedAction: "OPEN ORGANIC BIN",
        hardwareCommand: "organic", color: "#10b981"
    },
    INORGANIC: {
        id: "INORGANIC", name: "Inorganic", bin: "INORGANIC BIN",
        physicalBin: "inorganic", recommendedAction: "OPEN INORGANIC BIN",
        hardwareCommand: "inorganic", color: "#f59e0b"
    },
    REUSABLE: {
        id: "REUSABLE", name: "Reusable", bin: "REUSABLE BIN",
        physicalBin: "reusable", recommendedAction: "OPEN REUSABLE BIN",
        hardwareCommand: "reusable", color: "#3b82f6"
    }
};

const MODEL_CLASS_KEYS = ["ORGANIC", "INORGANIC", "REUSABLE"];
const MODEL_URL = new URL("models/model.json", document.baseURI).toString();
const CONFIDENCE_THRESHOLD = 0.80;
let customTfjsModel = null;
let modelLoadError = null;

async function loadAIClassifierModel() {
    if (customTfjsModel) return { custom: customTfjsModel };
    try {
        customTfjsModel = await tf.loadLayersModel(MODEL_URL);
        const outputShape = customTfjsModel.outputs[0].shape;
        if (outputShape[outputShape.length - 1] !== MODEL_CLASS_KEYS.length) {
            throw new Error(`Expected a 3-class model, received output shape ${JSON.stringify(outputShape)}.`);
        }
        modelLoadError = null;
        return { custom: customTfjsModel };
    } catch (error) {
        modelLoadError = error;
        customTfjsModel = null;
        console.error("[AI MODEL ERROR] Could not load the three-class model:", error);
        return { custom: null, error };
    }
}

async function classifyWasteItem(inputElement) {
    if (!customTfjsModel) await loadAIClassifierModel();
    if (!customTfjsModel) {
        return {
            success: false,
            error: `AI model unavailable at ${MODEL_URL}: ${modelLoadError ? modelLoadError.message : "models/model.json was not found."}`,
            predictedClass: "UNKNOWN", confidence: 0,
            probabilities: { ORGANIC: 0, INORGANIC: 0, REUSABLE: 0 },
            status: "MODEL ERROR"
        };
    }

    try {
        const tensor = tf.browser.fromPixels(inputElement)
            .resizeBilinear([224, 224]).toFloat().expandDims(0);
        const prediction = customTfjsModel.predict(tensor);
        const values = Array.from(await prediction.data());
        tensor.dispose();
        prediction.dispose();
        if (values.length !== MODEL_CLASS_KEYS.length) {
            throw new Error(`Expected 3 model probabilities, received ${values.length}.`);
        }

        const topIndex = values.reduce((best, value, index) => value > values[best] ? index : best, 0);
        const predictedClass = MODEL_CLASS_KEYS[topIndex];
        const confidence = values[topIndex] || 0;
        const category = WASTE_CATEGORIES[predictedClass];
        return {
            success: true, predictedClass, categoryName: category.name, confidence,
            isHighConfidence: confidence >= CONFIDENCE_THRESHOLD,
            status: confidence >= CONFIDENCE_THRESHOLD ? "CONFIRMED" : "LOW CONFIDENCE",
            message: confidence >= CONFIDENCE_THRESHOLD ? "Classification confirmed." : "Please show the waste item clearly.",
            recommendedBin: category.bin, recommendedAction: category.recommendedAction,
            hardwareCommand: category.hardwareCommand, physicalBin: category.physicalBin,
            themeColor: category.color,
            probabilities: {
                ORGANIC: values[0], INORGANIC: values[1], REUSABLE: values[2]
            }
        };
    } catch (error) {
        console.error("[AI INFERENCE ERROR]:", error);
        return {
            success: false, error: error.message, predictedClass: "UNKNOWN", confidence: 0,
            probabilities: { ORGANIC: 0, INORGANIC: 0, REUSABLE: 0 }, status: "MODEL ERROR"
        };
    }
}
