package com.smart.faceanalysis;

import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.dnn.Dnn;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FaceAnalysisPipeline {
    private static final String TAG = "FaceAnalysisPipeline";

    private static final float CONFIDENCE_THRESHOLD = 0.5f; // threshold for primary face detection, before starting analysis
    private static final float NMS_THRESHOLD = 0.42f; // threshold for overlapping bounding boxes


    // config for normalization
    // from https://huggingface.co/abhilash88/age-gender-prediction/blob/main/preprocessor_config.json
    private static final float[] AGE_GENDER_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] AGE_GENDER_STD = {0.229f, 0.224f, 0.225f};

    // config for normalization
    // from https://huggingface.co/abhilash88/face-emotion-detection/blob/main/preprocessor_config.json
    private static final float[] EMOTION_MEAN = {0.5f, 0.5f, 0.5f};
    private static final float[] EMOTION_STD = {0.5f, 0.5f, 0.5f};
    private static final float[] GRAYSCALE = {0.299f, 0.587f, 0.114f}; // COLOR_RGB2GRAY

    // dimensions need to match expected model inputs
    private static final int INPUT_DIMENSION_SHORT = 768;
    private static final int INPUT_DIMENSION_LONG = 1024;
    private static final int ViT_SIZE = 224;

    // current orientation
    private int ORIENTATION;
    public int INPUT_WIDTH;
    public int INPUT_HEIGHT;


    // executorch models
    private Module yoloModule; // points to either one below
    private final Module portraitModule;
    private final Module landscapeModule;
    private final Module ageGenderModule;
    private final Module emotionsModule;

    // buffers for image conversion
    private final int[] inputPixels;
    private final float[] inputArray;
    private final FloatBuffer inputBuffer;
    private long[] inputShape;
    private final int[] facePixels;
    private final float[] faceArray;
    private final FloatBuffer faceBuffer;
    private final long[] faceShape;

    // OpenCV Mats for NMS
    private Rect2d[] rectArray;
    private final MatOfRect2d boxes;
    private final MatOfFloat scores;
    private final MatOfInt indices;

    // track faces in a TRACKING_RESOLUTION x TRACKING_RESOLUTION grid to reduce number of analyses
    private int currentFrame = 0;
    private static final int TRACKING_RESOLUTION = 4;
    private final Map<Integer, FaceAnalysis> faceTracker = new HashMap<>();


    public FaceAnalysisPipeline(Module portraitModule, Module landscapeModule, Module ageGenderModule, Module emotionsModule,
                                int orientation) {

        this.portraitModule = portraitModule;
        this.landscapeModule = landscapeModule;
        this.ageGenderModule = ageGenderModule;
        this.emotionsModule = emotionsModule;

        updateOrientation(orientation);

        this.inputPixels = new int[INPUT_WIDTH * INPUT_HEIGHT];
        this.inputArray = new float[3 * INPUT_WIDTH * INPUT_HEIGHT];
        this.inputBuffer = Tensor.allocateFloatBuffer(3 * INPUT_WIDTH * INPUT_HEIGHT);

        this.facePixels = new int[ViT_SIZE * ViT_SIZE];
        this.faceArray = new float[3 * ViT_SIZE * ViT_SIZE];
        this.faceBuffer = Tensor.allocateFloatBuffer(3 * ViT_SIZE * ViT_SIZE);
        this.faceShape = new long[]{1, 3, ViT_SIZE, ViT_SIZE};

        this.boxes = new MatOfRect2d();
        this.scores = new MatOfFloat();
        this.indices = new MatOfInt();
    }

    /**
     * Change YOLO module and expected input dimension.
     * @param orientation android.content.res.Configuration.ORIENTATION_PORTRAIT or
     *                    android.content.res.Configuration.ORIENTATION_LANDSCAPE
     */
    public void updateOrientation(Integer orientation) {

        if (ORIENTATION == orientation) return;

        ORIENTATION = orientation;

        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
            INPUT_WIDTH = INPUT_DIMENSION_LONG;
            INPUT_HEIGHT = INPUT_DIMENSION_SHORT;

            yoloModule = landscapeModule;
        } else {
            INPUT_WIDTH = INPUT_DIMENSION_SHORT;
            INPUT_HEIGHT = INPUT_DIMENSION_LONG;

            yoloModule = portraitModule;
        }

        inputShape = new long[]{1, 3, INPUT_HEIGHT, INPUT_WIDTH};
    }

    /**
     * Run executorch modules with sample input
     */
    public void warmUp() {
        portraitModule.forward();
        landscapeModule.forward();
        ageGenderModule.forward();
        emotionsModule.forward();
    }

    /**
     * Detect faces in the given bitmap
     * @param bitmap The bitmap to analyze.
     *               Needs to have the same dimension as INPUT_WIDTH / INPUT_HEIGHT.
     * @param trackFaces Track face locations, for averaged probabilities in live view
     * @return List of {@link FaceAnalysis} with detected bounding boxes
     * and previous predictions, if tracking is activated
     */
    public List<FaceAnalysis> detectFaces(Bitmap bitmap, Boolean trackFaces) {
        List<FaceAnalysis> results = new ArrayList<>();

        try {

            if (BuildConfig.ENABLE_LOGGING) {
                Log.d(TAG, "Received bitmap");
            }

            // validate input
            Tensor yoloInput = preprocessInput(bitmap);
            if (yoloInput == null) {
                return results;
            }

            if (BuildConfig.ENABLE_LOGGING) {
                Log.d(TAG, "Preprocessed bitmap");
            }

            // get raw YOLO detections
            currentFrame++;

            EValue[] yoloOutput = yoloModule.forward(EValue.from(yoloInput));
            if (yoloOutput == null || yoloOutput.length == 0) {
                faceTracker.clear();
                return results;
            }

            if (BuildConfig.ENABLE_LOGGING) {
                Log.d(TAG, "Predicted faces");
            }

            // perform NMS
            List<Rect> detections = validateDetections(yoloOutput);

            if (BuildConfig.ENABLE_LOGGING) {
                Log.d(TAG, "Valid detections: " + detections.size());
            }

            // sort to draw in order
            detections.sort(Comparator.comparingInt(Rect::centerX));

            // get faces
            for (Rect detection : detections) {

                FaceAnalysis face;
                if (trackFaces) {
                    int loc = getFaceLocation(detection);
                    face = faceTracker.computeIfAbsent(loc, k -> new FaceAnalysis());
                } else {
                    face = new FaceAnalysis();
                }

                // update bounding box
                face.setBoundingBox(currentFrame, detection);

                results.add(face);
            }

            // remove untracked faces
            faceTracker.entrySet().removeIf(entry -> entry.getValue().lastFrameTracked < currentFrame);

        } catch (Exception e) {
            Log.e(TAG, "Face detection failed: ", e);
        }

        return results;
    }

    /**
     * Predicts age, gender and/or facial emotion for a given face.
     * (Have to analyze one by one, because executorch doesn't handle batch size > 1 well)
     * @param bitmap Bitmap used in detectFaces
     * @param face Face to analyze
     */
    public void analyzeFace(Bitmap bitmap, FaceAnalysis face, boolean analyzeAge, boolean analyzeEmotion) {

        try
        {
            float ratioX = (float)ViT_SIZE / face.boundingBox.width();
            float ratioY = (float)ViT_SIZE / face.boundingBox.height();
            float middle = ViT_SIZE / 2.0f;
            Matrix scaleMatrix = new Matrix();
            scaleMatrix.setScale(ratioX, ratioY, middle, middle);

            Bitmap croppedFace = Bitmap.createBitmap(bitmap,
                    face.boundingBox.left, face.boundingBox.top, face.boundingBox.width(), face.boundingBox.height(),
                    scaleMatrix, true);

            // pixel values are reused for both predictions
            croppedFace.getPixels(facePixels, 0, ViT_SIZE, 0, 0, ViT_SIZE, ViT_SIZE);


            if (BuildConfig.ENABLE_LOGGING) {
                Log.d(TAG, "Cropped face");
            }

            // predict age and gender
            if (analyzeAge) {

                float[] probabilities = predict(ageGenderModule, AGE_GENDER_MEAN, AGE_GENDER_STD, false);

                if (probabilities != null) {
                    face.setAgeGender(Math.round(probabilities[0]), probabilities[1]);

                    if (BuildConfig.ENABLE_LOGGING) {
                        Log.d(TAG, "predicted gender & age");
                    }
                }
            }

            //predict emotion
            if (analyzeEmotion) {

                float[] probabilities = predict(emotionsModule, EMOTION_MEAN, EMOTION_STD, true);

                if (probabilities != null) {
                    face.setEmotion(probabilities);

                    if (BuildConfig.ENABLE_LOGGING) {
                        Log.d(TAG, "predicted emotion");
                    }
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "Failed to analyze face: ", e);
        }
    }

    @NonNull
    private Integer getFaceLocation(@NonNull Rect box) {
        // turns the image into a TRACKING_RESOLUTION x TRACKING_RESOLUTION grid
        // and returns the index of the cell containing
        // the center of the bounding box
        return (box.centerX() / (INPUT_WIDTH / TRACKING_RESOLUTION)) * TRACKING_RESOLUTION
             + (box.centerY() / (INPUT_HEIGHT / TRACKING_RESOLUTION));
    }

    @Nullable
    private Tensor preprocessInput(Bitmap bitmap) {

        try {
            if (bitmap == null || bitmap.isRecycled()) {
                Log.w(TAG, "Invalid bitmap for preprocessing");
                return null;
            }

            // for ARGB8888, getPixels() + FloatBuffer.put(float[]) performed faster than
            // any combination of copyPixelsToBuffer(), FloatBuffer.put(float) or OpenCV Mat
            bitmap.getPixels(inputPixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);

            // convert to RGB
            final int offsetG = INPUT_WIDTH * INPUT_HEIGHT;
            final int offsetB = 2 * INPUT_WIDTH * INPUT_HEIGHT;

            for (int i = 0; i < inputPixels.length; i++) {
                int pixel = inputPixels[i];
                inputArray[i] = (((pixel >> 16) & 0xff) / 255.0f); // R
                inputArray[i + offsetG] = (((pixel >> 8) & 0xff) / 255.0f); // G
                inputArray[i + offsetB] = ((pixel & 0xff) / 255.0f); // B
            }

            // create tensor from RGB values with required dimensions
            inputBuffer.rewind();
            inputBuffer.put(inputArray);

            return Tensor.fromBlob(inputBuffer, inputShape);

        } catch (Exception e) {
            Log.e(TAG, "Failed to preprocess input", e);
            return null;
        }
    }

    @Nullable
    private Tensor preprocessFace(float[] mean, float[] std, boolean grayscale) {
        try {

            // uses pixels set in analyzeFace()

            // convert to RGB & normalize
            final int offsetG = ViT_SIZE * ViT_SIZE;
            final int offsetB = 2 * ViT_SIZE * ViT_SIZE;

            if (grayscale) {
                for (int i = 0; i < facePixels.length; i++) {
                    int pixel = facePixels[i];
                    float r = ((pixel >> 16) & 0xff) / 255.0f * GRAYSCALE[0];
                    float g = ((pixel >> 8) & 0xff) / 255.0f * GRAYSCALE[1];
                    float b = (pixel & 0xff) / 255.0f * GRAYSCALE[2];

                    float gray = ((r + g + b) - mean[0]) / std[0];
                    faceArray[i] = gray; // R
                    faceArray[i + offsetG] = gray; // G
                    faceArray[i + offsetB] = gray; // B
                }
            } else {
                for (int i = 0; i < facePixels.length; i++) {
                    int pixel = facePixels[i];
                    faceArray[i] = (((pixel >> 16) & 0xff) / 255.0f - mean[0]) / std[0]; // R
                    faceArray[i + offsetG] = (((pixel >> 8) & 0xff) / 255.0f - mean[1]) / std[1]; // G
                    faceArray[i + offsetB] = ((pixel & 0xff) / 255.0f - mean[2]) / std[2]; // B
                }
            }

            // create tensor from RGB values with required dimensions
            faceBuffer.rewind();
            faceBuffer.put(faceArray);

            return Tensor.fromBlob(faceBuffer, faceShape);
        } catch (Exception e) {
            Log.e(TAG, "Failed to preprocess face", e);
            return null;
        }
    }

    @NonNull
    private List<Rect> validateDetections(EValue[] outputs) {
        List<Rect> detections = new ArrayList<>();

        try {

            // Parse output of optimized YOLOModel.
            // Will not work with standard YOLO model.
            float[] coordinates = outputs[0].toTensor().getDataAsFloatArray();
            float[] confidences = outputs[1].toTensor().getDataAsFloatArray();


            // number of YOLO anchors depends on input dimension
            final int anchors = coordinates.length / 3;

            if (rectArray == null){
                rectArray = new Rect2d[anchors];
            }

            // convert to OpenCV Mat types
            for (int i = 0; i < anchors; i++) {

                rectArray[i] = new Rect2d(coordinates[i],
                        coordinates[anchors + i],
                        coordinates[2 * anchors + i],
                        coordinates[2 * anchors + i]);
            }

            boxes.fromArray(rectArray);
            scores.fromArray(confidences);


            // Apply Non-Maximum Suppression.
            // Be aware that top_k is applied *before* nms_threshold, so best to leave at 0.
            Dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices, 1, 0);


            int[] indicesArray = new int[(int) (indices.total() * indices.channels())];
            indices.get(0, 0, indicesArray);

            for (int i : indicesArray) {

                Rect2d rect = rectArray[i];

                double left = rect.x;
//                // Uncomment to fix misaligned output for dynamic model
//                // when image ratio doesn't fit to quantization ratio.
//                // Assumes model sample input was portrait.
//                // Will NOT fix missing detections at the right border of the image.
//                if (ORIENTATION == Configuration.ORIENTATION_LANDSCAPE) {
//                    double center = rect.x + (rect.width / 2);
//                    center = ((center - (INPUT_WIDTH / 2f)) / 0.96) + (INPUT_WIDTH / 2f);
//                    left = center - (rect.width / 2);
//                } else {
//                    left = rect.x;
//                }

                // clamp to image bounds
                int top = Math.max(0, (int)rect.y);
                int right = Math.min(INPUT_WIDTH, (int)(left + rect.width));
                int bottom = Math.min(INPUT_HEIGHT, (int)(rect.y + rect.height));

                detections.add(new Rect(Math.max(0, (int)left), top, right, bottom));
            }

        } catch (Exception e) {
            Log.e(TAG, "Failed to parse YOLO output", e);
        }

        return detections;
    }

    private float[] predict(Module module, float[] mean, float[] std, boolean grayscale) {

        // get predictions from ViT modules

        float[] probabilities = null;

        Tensor classifierInput = preprocessFace(mean, std, grayscale);
        if (classifierInput != null) {

            if (BuildConfig.ENABLE_LOGGING) {
                Log.d(TAG, "Normalized face");
            }

            EValue inputEValue = EValue.from(classifierInput);
            EValue[] classifierOutputs = module.forward(inputEValue);
            probabilities = classifierOutputs[0].toTensor().getDataAsFloatArray();
        }

        return probabilities;
    }

    public void release(){
        boxes.release();
        scores.release();
        indices.release();

        if (yoloModule != null) {
            yoloModule.destroy();
        }
        if (ageGenderModule != null) {
            ageGenderModule.destroy();
        }
        if (emotionsModule != null) {
            emotionsModule.destroy();
        }
    }
}