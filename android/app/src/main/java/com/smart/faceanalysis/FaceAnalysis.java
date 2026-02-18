package com.smart.faceanalysis;

import android.graphics.Rect;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class FaceAnalysis {

    private static final int ANALYZE_AGE_RATE = 10; // frames between age & gender analysis
    private static final int ANALYZE_EMOTION_RATE = 5; // frames between emotion analysis
    private static final String[] EMOTIONS =
            { "angry", "disgusted", "afraid", "happy", "sad", "surprised", "neutral" };


    /**
     * Detected square bounding box around face
     */
    public Rect boundingBox;

    /**
     * Calculates the average of all predicted ages
     * @return Age as integer between 1 and 100
     */
    public Integer getAge() {
        float sum = 0;
        for (int a:ages) {
            sum += a;
        }
        return (int)(sum / ages.size());
    }
    private final List<Integer> ages = new ArrayList<>();

    /**
     * Calculates the average of all predicted genders
     * @return Gender as string - either "female" or "male"
     */
    public String getGender() {
        if (genderProbability() >= 0.5) {
            return "female";
        }
        return "male";
    }
    private float genderProbability() {
        float sum = 0;
        for (float prob:genderProbabilities) {
            sum += prob;
        }
        return sum / genderProbabilities.size();
    }
    private final List<Float> genderProbabilities = new ArrayList<>();

    /**
     * Last predicted emotion -
     * one of "angry", "disgusted", "afraid", "happy", "sad", "surprised", "neutral"
     * */
    public String emotion;
    private float emotionConfidence;

    /**
     * Last frame the face was detected
     * (used to clear from face tracker)
     */
    public int lastFrameTracked = 0;

    // immediate age & gender analysis
    private int nextAgeAnalysis = 1;
    public Boolean needsAgeAnalysis(){
        return nextAgeAnalysis == 0;
    }

    // first emotion analysis after 1 frame
    // for drawing bounding box + age & gender result
    private int nextEmotionAnalysis = 2;
    public Boolean needsEmotionAnalysis(){
        return nextEmotionAnalysis == 0;
    }


    /**
     * Sets the current bounding box for the detected face
     * and reduces frame time until next analyses
     * @param frame The frame of detection
     * @param boundingBox The detected boundingBox
     */
    public void setBoundingBox(Integer frame, Rect boundingBox) {
        this.lastFrameTracked = frame;
        this.boundingBox = boundingBox;

        this.nextAgeAnalysis--;
        this.nextEmotionAnalysis--;
    }

    /**
     * Adds the predicted age & gender to the list
     * and resets frame time until next analysis to {@value ANALYZE_AGE_RATE}
     * @param age The predicted age in years
     * @param genderProbability The probability that gender is female
     */
    public void setAgeGender(Integer age, float genderProbability) {

        this.ages.add(Math.max(0, Math.min(100, age)));
        this.genderProbabilities.add(genderProbability);

        this.nextAgeAnalysis = ANALYZE_AGE_RATE;
    }

    /**
     * Sets the emotion & emotion confidence
     * and resets frame time until next analysis to {@value ANALYZE_EMOTION_RATE}
     * @param probabilities Predicted probabilities for each emotion.
     *                      Confidence will be set as softmax.
     */
    public void setEmotion(float[] probabilities) {

        int maxIndex = -1;
        float probSum = 0;
        float maxProb = 0;

        for (int i = 1; i < probabilities.length; i++) {
            float prob = (float)Math.exp(probabilities[i]);
            probSum += prob;

            if (prob > maxProb) {
                maxProb = prob;
                maxIndex = i;
            }
        }

        this.emotion = (maxIndex > 0) ?  EMOTIONS[maxIndex] : "neutral";
        this.emotionConfidence = (probSum > 0) ? maxProb / probSum : 0; // softmax

        this.nextEmotionAnalysis = ANALYZE_EMOTION_RATE;
    }

    /**
     * Formatted brief analysis, to use as label
     * @return Age group ("young", "adult" or "elderly")
     * plus predicted gender ("male" or "female")
     */
    public String getLabel() {
        if (ages.isEmpty()) {
            return null;
        }

        // age groups based on https://doi.org/10.1101/2025.01.10.25320339
        // 1 to 4, 5 to 8, 9 to 17, 18 to 33, 34 to 50, 51 to 67, 68 to 89, 90+
        Integer age = getAge();
        String ageGroup;
        if (age < 34){
            ageGroup = "young";
        } else if (age > 67) {
            ageGroup = "elderly";
        } else {
            ageGroup = "adult";
        }

        return ageGroup + " " + getGender();
    }

    /**
     * Formatted analysis
     * @return if emotion has not been predicted:
     * "gender: {gender}, age: {age}, ({genderConfidence})"
     * else:
     * "gender: {gender}, age: {age}, emotion: {emotion} ({combined confidence})"
     */
    public String getAnalysis() {

        if (ages.isEmpty()) {
            return "• analyzing...";
        }

        String gender;
        float genderConfidence = genderProbability();
        if (genderConfidence < 0.5){
            gender = "male";
            genderConfidence = 1 - genderConfidence;
        } else{
            gender = "female";
        }

        if (emotionConfidence == 0) {
            return String.format(Locale.ENGLISH,"• gender: %s, age: %s (%.1f%%)",
                    gender, getAge(), genderConfidence * 100);
        }

        return String.format(Locale.ENGLISH,"• gender: %s, age: %s, emotion: %s (%.1f%%)",
                gender, getAge(), emotion, genderConfidence * emotionConfidence * 100);
    }
}