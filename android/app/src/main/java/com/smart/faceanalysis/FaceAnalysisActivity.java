package com.smart.faceanalysis;

import android.Manifest;
import android.content.ContentResolver;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.pytorch.executorch.Module;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;


public class FaceAnalysisActivity extends AppCompatActivity {
    private static final String TAG = "FaceAnalysisActivity";
    private static final int REQUEST_CODE_PERMISSIONS = 1001;
    private static final String[] REQUIRED_PERMISSIONS = {Manifest.permission.CAMERA};

    private Button sessionButton; // start & stop camera detection
    private Button imageButton; // start image detection
    private PreviewView previewView; // shows current camera proxy
    private View fadeView; // fade camera in and out
    private ImageView photoView; // shows selected photo
    private ImageView overlayView; // displays bounding boxes

    // buffer for bounding boxes
    private Canvas overlayCanvas;
    private Bitmap overlayBitmap;

    private TextView resultTextView; // displays analysis results

    // calculated ratio between analysis image size and canvas size
    // automatically set on detection
    private float scaleWidth = 1;
    private float scaleHeight = 1;

    // style
    private Paint boxPaint;
    private Paint textPaint;
    private Paint backgroundPaint;

    private ExecutorService analysisExecutor;
    private ProcessCameraProvider cameraProvider;
    private FaceAnalysisPipeline facePipeline;
    private final AtomicBoolean isProcessing = new AtomicBoolean(false);
    private boolean isSessionActive = false;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        initializeViews();
        setStyles();
        updateButtons();
        setupButtons();

        // run image analysis on separate thread
        analysisExecutor = Executors.newSingleThreadExecutor();

        // load OpenCV library
        if (!OpenCVLoader.initLocal()) {
            Log.e("OpenCV", "Failed to load OpenCV library");
            Toast.makeText(this, "Failed to load OpenCV library.", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        try {
            // load executorch models
            int orientation = getResources().getConfiguration().orientation;
            Module portraitModule = getYoloModule(Configuration.ORIENTATION_PORTRAIT);
            Module landscapeModule = getYoloModule(Configuration.ORIENTATION_LANDSCAPE);

            String classifierPath = assetFilePath("age_gender.pte");
            String emotionPath = assetFilePath("emotion.pte");

            Module classifierModule = Module.load(classifierPath);
            Module emotionsModule = Module.load(emotionPath);

            // initialize pipeline
            facePipeline = new FaceAnalysisPipeline(portraitModule, landscapeModule, classifierModule, emotionsModule,
                    orientation);

        } catch (Exception e) {
            Log.e(TAG, "Failed to load models", e);
            Toast.makeText(this, "Failed to load models: " + e.getMessage(), Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        // warmup executorch & vulkan
        new Thread(() -> facePipeline.warmUp()).start();

        // get camera permission
        if (missingPermission()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    @NonNull
    private Module getYoloModule(int orientation) throws IOException {

        String modulePath;
        try {
            // dynamic model
            modulePath = assetFilePath("face_detector.pte");
        } catch (IOException e) {
            // separate models
            if (orientation == Configuration.ORIENTATION_PORTRAIT){
                modulePath = assetFilePath("face_detector_portrait.pte");
            } else {
                modulePath = assetFilePath("face_detector_landscape.pte");
            }
        }
        return Module.load(modulePath);
    }

    private void initializeViews() {
        setContentView(R.layout.main_activity);

        previewView = findViewById(R.id.previewView);
        fadeView = findViewById(R.id.fadeView);
        photoView = findViewById(R.id.photoView);
        overlayView = findViewById(R.id.overlayView);
        resultTextView = findViewById(R.id.resultTextView);
        sessionButton = findViewById(R.id.sessionButton);
        imageButton = findViewById(R.id.imageButton);
    }

    private void setupButtons() {
        sessionButton.setOnClickListener(v -> {
            if (isSessionActive) {
                endCameraSession();
            } else {
                startCameraSession();
            }
        });

        // register photo picker activity launcher
        ActivityResultLauncher<PickVisualMediaRequest> pickMedia =
                registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), uri -> {
                    if (uri != null) {
                        Log.d("PhotoPicker", "Selected URI: " + uri);

                        clearOverlay();

                        // analyze selected image
                        ContentResolver resolver = this.getContentResolver();
                        new Thread(() -> {
                            ImageDecoder.Source source = ImageDecoder.createSource(resolver, uri);
                            try {
                                Bitmap bitmap = ImageDecoder.decodeBitmap(source);

                                analyzeImage(bitmap);
                            } catch (IOException e) {
                                Log.e(TAG, "Failed to load image: ", e);
                            }
                        }).start();
                    }
                });

        PickVisualMediaRequest.Builder imagePicker = new PickVisualMediaRequest.Builder()
                .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE);

        imageButton.setOnClickListener(v -> {
            if (!isProcessing.get()) {
                pickMedia.launch(imagePicker.build());
        }});
    }

    private void startCameraSession() {

        imageButton.setEnabled(false);

        startCamera();

        clearOverlay();
        prepareCanvas(previewView.getWidth(), previewView.getHeight(), true);
        facePipeline.updateOrientation(getResources().getConfiguration().orientation);

        sessionButton.setText(R.string.end_session);
        resultTextView.setText(R.string.session_started);
    }

    private void endCameraSession() {

        isSessionActive = false;

        stopCamera();

        resultTextView.setText(R.string.session_ended);

        clearOverlay();
        updateButtons();
    }

    private void clearOverlay() {
        photoView.setImageBitmap(null);
        overlayView.setImageBitmap(null);
    }

    private void updateButtons() {
        if (isSessionActive) {
            sessionButton.setText(R.string.end_session);
            imageButton.setEnabled(false);
        } else {
            sessionButton.setText(R.string.start_session);
            imageButton.setEnabled(true);
        }
    }
    private void setStyles() {

        boxPaint = new Paint();
        boxPaint.setColor(Color.rgb(73,93,146));
        boxPaint.setStrokeWidth(5);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setAntiAlias(true);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(30);
        textPaint.setAntiAlias(true);
        textPaint.setShadowLayer(4, 2, 2, Color.BLACK);

        backgroundPaint = new Paint();
        backgroundPaint.setColor(Color.rgb(73,93,146));
    }

    @NonNull
    private String assetFilePath(String assetName) throws IOException {
        // Creates a temporary copy of model files in cache.
        // Workaround to get absolute file path of asset, as required by executorch.Module.load().

        File file = new File(this.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = this.getAssets().open(assetName);
             FileOutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4096];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
        }
        return file.getAbsolutePath();
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindPreviewAndAnalyzer(cameraProvider);

                // fade in camera view
                fadeView.postDelayed(() -> {
                    fadeView.animate()
                        .alpha(0.0f)
                        .setDuration(800)
                        .setListener(null);

                    isSessionActive = true;
                    }, 1000);

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Camera initialization failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void stopCamera() {

        if (cameraProvider != null) {

            // fade out camera view
            fadeView.animate()
                    .alpha(1.0f)
                    .setDuration(800)
                    .setListener(null);

            cameraProvider.unbindAll();
        }
    }

    private void bindPreviewAndAnalyzer(@NonNull ProcessCameraProvider cameraProvider) {

        // try to match image proxy dimensions with analysis dimensions
        ResolutionStrategy resolutionStrategy = new ResolutionStrategy(new Size(facePipeline.INPUT_WIDTH,facePipeline.INPUT_HEIGHT)
                , ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER);

        ResolutionSelector resolutionSelector = new ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(resolutionStrategy)
                .build();

        Preview preview = new Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageRotationEnabled(true)
                .build();

        imageAnalysis.setAnalyzer(analysisExecutor, this::analyzeImage);

        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis);

            if (BuildConfig.ENABLE_LOGGING) {
                Log.d(TAG, "Camera bound successfully");
            }

        } catch (Exception e) {
            Log.e(TAG, "Camera binding failed: ", e);
        }
    }

    private void prepareCanvas(int width, int height, boolean scale) {
        // clear canvas
        overlayCanvas = null;

        // create empty bitmap for canvas to draw on
        overlayBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888, true);

        if (scale && (width != facePipeline.INPUT_WIDTH || height != facePipeline.INPUT_HEIGHT)) {
            // scale if view dimensions differ from analysis dimensions
            scaleWidth = (float) width / (float) facePipeline.INPUT_WIDTH;
            scaleHeight = (float) height / (float) facePipeline.INPUT_HEIGHT;
        } else {
            scaleWidth = 1;
            scaleHeight = 1;
        }
    }

    private Bitmap scaleWithRatio(@NonNull Bitmap bitmap, int maxWidth, int maxHeight) {

        int width, height;
        float widthRatio = (float)bitmap.getWidth() / maxWidth;
        float heightRatio = (float)bitmap.getHeight() / maxHeight;

        if (widthRatio >= heightRatio) {
            width = maxWidth;
            height = (int)(((float)width / bitmap.getWidth()) * bitmap.getHeight());
        } else {
            height = maxHeight;
            width = (int)(((float)height / bitmap.getHeight()) * bitmap.getWidth());
        }
        Bitmap scaledBitmap = Bitmap.createBitmap(maxWidth, maxHeight, Bitmap.Config.ARGB_8888, false);

        float ratioX = (float)width / bitmap.getWidth();
        float ratioY = (float)height / bitmap.getHeight();
        float middleX = maxWidth / 2.0f;
        float middleY = maxHeight / 2.0f;
        Matrix scaleMatrix = new Matrix();
        scaleMatrix.setScale(ratioX, ratioY, middleX, middleY);

        Canvas canvas = new Canvas(scaledBitmap);
        canvas.drawColor(Color.WHITE);
        canvas.setMatrix(scaleMatrix);
        canvas.drawBitmap(bitmap.copy(Bitmap.Config.ARGB_8888, true),
                middleX - bitmap.getWidth() / 2f, middleY - bitmap.getHeight() / 2f, new Paint(Paint.FILTER_BITMAP_FLAG));
        return scaledBitmap;
    }

    /**
     * Analyze static image.
     * Pads and scales image, analyzes age, gender, and emotion for all detected faces,
     * then outputs the results onto the overlay.
     * @param bitmap Image to analyze.
     *               Can have arbitrary aspect ratio.
     */
    private void analyzeImage(@NonNull Bitmap bitmap) {
        try {

            if (BuildConfig.ENABLE_LOGGING) {
                Log.d(TAG, "Start processing.");
            }

            isProcessing.set(true);

            // scale & pad bitmap to aspect ratio
            if (bitmap.getWidth() > bitmap.getHeight()) {
                facePipeline.updateOrientation(Configuration.ORIENTATION_LANDSCAPE);
            } else {
                facePipeline.updateOrientation(Configuration.ORIENTATION_PORTRAIT);
            }

            Bitmap scaledBitmap = scaleWithRatio(bitmap, facePipeline.INPUT_WIDTH, facePipeline.INPUT_HEIGHT);

            // show image and prepare overlay
            runOnUiThread(() -> {
                photoView.setImageBitmap(scaledBitmap);
                prepareCanvas(facePipeline.INPUT_WIDTH, facePipeline.INPUT_HEIGHT, false);
            });


            // detect faces
            List<FaceAnalysis> detections = facePipeline.detectFaces(scaledBitmap, false);

            runOnUiThread(() -> {
                // show bounding boxes
                draw(detections, false);
            });


            // analyze faces
            for (FaceAnalysis face : detections) {

                facePipeline.analyzeFace(scaledBitmap, face, true, true);

                runOnUiThread(() -> {
                    // show analysis result
                    draw(detections, true);
                });
            }

        } catch (Exception e) {
            Log.e(TAG, "Analysis failed: ", e);
        } finally {
            isProcessing.set(false);
        }
    }

    /**
     * Analyze image from camera view.
     * Scales image, detects faces and possibly analyzes age, gender, and / or emotion,
     * then outputs the results onto the overlay.
     * To get a full analysis, has to be called for at least two consecutive frames.
     * @param imageProxy Camera view.
     *                   Needs to have 3:4 or 4:3 aspect ratio.
     */
    private void analyzeImage(@NonNull ImageProxy imageProxy) {
        try (imageProxy) {

            // Skip if already processing or session not active
            if (isProcessing.get() || !isSessionActive) {
                return;
            }

            if (BuildConfig.ENABLE_LOGGING) {
                Log.d(TAG, "Start processing.");
            }

            isProcessing.set(true);

            // create scaled bitmap
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(imageProxy.toBitmap(), facePipeline.INPUT_WIDTH, facePipeline.INPUT_HEIGHT, true);

            // detect and track faces
            List<FaceAnalysis> detections = facePipeline.detectFaces(scaledBitmap, true);

            if (!isSessionActive) return;

            runOnUiThread(() -> {
                // show bounding boxes
                draw(detections, false);
            });

            // analyze faces
            for (FaceAnalysis face : detections) {

                // only analyze face every couple frames to increase throughput
                Boolean analyzeAge = face.needsAgeAnalysis();
                Boolean analyzeEmotion = face.needsEmotionAnalysis();
                if (analyzeAge || analyzeEmotion) {
                    facePipeline.analyzeFace(scaledBitmap, face, analyzeAge, analyzeEmotion);

                    if (!isSessionActive) break;

                    runOnUiThread(() -> {
                        // show analysis result
                        draw(detections, true);
                    });
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "Analysis failed: ", e);
        } finally {
            isProcessing.set(false);
        }
    }

    private void draw(@NonNull List<FaceAnalysis> detections, Boolean isUpdate) {

        if (detections.isEmpty()) {
            overlayView.setImageBitmap(null);
            resultTextView.setText(R.string.no_detections);
            return;
        }

        if (overlayBitmap == null) {
            // create empty bitmap for canvas to draw on
            int width = previewView.getWidth();
            int height = previewView.getHeight();

            overlayBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888, true);

            if (width != facePipeline.INPUT_WIDTH || height != facePipeline.INPUT_HEIGHT) {
                // scale if view dimensions differ from analysis dimensions
                scaleWidth = (float) width / (float) facePipeline.INPUT_WIDTH;
                scaleHeight = (float) height / (float) facePipeline.INPUT_HEIGHT;
            }
        }

        if (overlayCanvas == null) {
            overlayCanvas = new Canvas(overlayBitmap);
        } else if (!isUpdate) {
            // clear canvas
            overlayCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
        }

        StringBuilder resultText = new StringBuilder("Faces detected:\n");

        for (FaceAnalysis detection : detections) {

            Rect boundingBox;
            if (scaleWidth != 1 || scaleHeight != 1) {

                boundingBox = new Rect((int)(detection.boundingBox.left * scaleWidth),
                        (int)(detection.boundingBox.top * scaleHeight),
                        (int)(detection.boundingBox.right * scaleWidth),
                        (int)(detection.boundingBox.bottom * scaleHeight));
            } else {
                boundingBox = detection.boundingBox;
            }

            if (!isUpdate) {
                // draw bounding box
                overlayCanvas.drawRect(boundingBox, boxPaint);
            }

            String label = detection.getLabel();
            if (label != null) {

                // draw background for label
                float textWidth = textPaint.measureText(label);

                overlayCanvas.drawRect(boundingBox.left - 2, boundingBox.bottom - 3,
                        boundingBox.left - 2 + Math.max(textWidth + 20, boundingBox.width() + 5),
                        boundingBox.bottom + 32, backgroundPaint);

                // draw label
                overlayCanvas.drawText(label,
                        boundingBox.left + 5,
                        boundingBox.bottom + 24,
                        textPaint);
            }

            // show analysis result
            resultText.append(detection.getAnalysis()).append("\n");
        }

        overlayView.setImageBitmap(overlayBitmap);
        resultTextView.setText(resultText.toString());
    }

    private boolean missingPermission() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return true;
            }
        }
        return false;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (missingPermission()) {
                Toast.makeText(this, "Permissions not granted", Toast.LENGTH_SHORT).show();
                sessionButton.setEnabled(false);
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        if (analysisExecutor != null) {
            analysisExecutor.shutdown();
        }
        if (facePipeline != null) {
            facePipeline.release();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        updateButtons();
    }
}