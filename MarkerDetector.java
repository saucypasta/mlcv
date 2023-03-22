package org.firstinspires.ftc.teamcode;

import android.graphics.Bitmap;
import android.util.Log;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;
import java.util.Iterator;

public class MyOpenCvPipeline extends OpenCvPipeline {
    private static final String  TAG = "HHS:MarkerDetector: ";
    // Lower and Upper bounds for range checking in HSV color space
    private Scalar mLowerBound = new Scalar(0);
    private Scalar mUpperBound = new Scalar(0);
    // Minimum contour area in percent for contours filtering
    private static double mMinContourArea = 0.1;
    // Color radius for range checking in HSV color space
    private Scalar mColorRadius = new Scalar(40,47,80,0);  // 20
    private Mat mSpectrum = new Mat();
    private List<MatOfPoint> mContours = new ArrayList<MatOfPoint>();
    private Scalar mBlobColorHsv;
    private int markers;
    Bitmap bmp;
    private boolean countMarkers = false;
    private boolean countMarkersReady = false;

    // Cache
    Mat mPyrDownMat = new Mat();
    Mat mHsvMat = new Mat();
    Mat mMask = new Mat();
    Mat mDilatedMask = new Mat();
    Mat mHierarchy = new Mat();
    private Mat resizeImageOutput = new Mat();
    private Mat hsvThresholdOutput = new Mat();
    private Mat cvErodeOutput = new Mat();
    private Mat cvDilateOutput = new Mat();
    private ArrayList<MatOfPoint> findContoursOutput = new ArrayList<MatOfPoint>();
    private ArrayList<MatOfPoint> filterContoursOutput = new ArrayList<MatOfPoint>();

    public void setColorRadius(Scalar radius) {
        mColorRadius = radius;
    }

    public void setHsvColor(Scalar hsvColor) {
        double minH = (hsvColor.val[0] >= mColorRadius.val[0]) ? hsvColor.val[0]-mColorRadius.val[0] : 0;
        double maxH = (hsvColor.val[0]+mColorRadius.val[0] <= 255) ? hsvColor.val[0]+mColorRadius.val[0] : 255;

        mLowerBound.val[0] = minH;
        mUpperBound.val[0] = maxH;

        mLowerBound.val[1] = hsvColor.val[1] - mColorRadius.val[1];
        mUpperBound.val[1] = hsvColor.val[1] + mColorRadius.val[1];

        mLowerBound.val[2] = hsvColor.val[2] - mColorRadius.val[2];
        mUpperBound.val[2] = hsvColor.val[2] + mColorRadius.val[2];

        mLowerBound.val[3] = 0;
        mUpperBound.val[3] = 255;

//        Mat spectrumHsv = new Mat(1, (int)(maxH-minH), CvType.CV_8UC3);
//
//        for (int j = 0; j < maxH-minH; j++) {
//            byte[] tmp = {(byte)(minH+j), (byte)255, (byte)255};
//            spectrumHsv.put(0, j, tmp);
//        }
//
//        Imgproc.cvtColor(spectrumHsv, mSpectrum, Imgproc.COLOR_HSV2RGB_FULL, 4);
    }

    public Mat getSpectrum() {
        return mSpectrum;
    }

    public void setMinContourArea(double area) {
        mMinContourArea = area;
    }

    /**
     * This is the primary method that runs the entire pipeline and updates the outputs.
     */
    public Mat processFrame(Mat source0) {
        if (countMarkers) {

            mBlobColorHsv = new Scalar(255);
            // Imgproc.pyrDown(source0, mPyrDownMat);
            // Imgproc.pyrDown(mPyrDownMat, mPyrDownMat);

            Imgproc.cvtColor(source0, mHsvMat, Imgproc.COLOR_RGB2HSV_FULL);

            //(24.8125, 239.234375, 201.5, 0.0)
            mBlobColorHsv.val[0] = 6; // 12
            mBlobColorHsv.val[1] = 208.0;
            mBlobColorHsv.val[2] = 127.0;
            mBlobColorHsv.val[3] = 0.0;

            setHsvColor(mBlobColorHsv);

            Core.inRange(mHsvMat, mLowerBound, mUpperBound, mMask);

            // Step CV_erode0:
            Mat cvErodeSrc = mMask;
            Mat cvErodeKernel = new Mat();
            Point cvErodeAnchor = new Point(-1, -1);
            int cvErodeIterations = 3;
            //int cvErodeIterations = 0;
            int cvErodeBordertype = Core.BORDER_CONSTANT;
            Scalar cvErodeBordervalue = new Scalar(-1);
            Imgproc.erode(cvErodeSrc, cvErodeOutput, cvErodeKernel, cvErodeAnchor, cvErodeIterations, cvErodeBordertype, cvErodeBordervalue);

            // Step CV_dilate0:
            Mat cvDilateSrc = cvErodeOutput;
            Mat cvDilateKernel = new Mat();
            Point cvDilateAnchor = new Point(-1, -1);
            int cvDilateIterations = 15;
            int cvDilateBordertype = Core.BORDER_CONSTANT;
            Scalar cvDilateBordervalue = new Scalar(-1);
            Imgproc.dilate(cvDilateSrc, mDilatedMask, cvDilateKernel, cvDilateAnchor, cvDilateIterations, cvDilateBordertype, cvDilateBordervalue);

            List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

            Imgproc.findContours(mDilatedMask, contours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // Find max contour area
            mContours.clear();
            double maxArea = 0;
            MatOfPoint maxContour = new MatOfPoint();
            Iterator<MatOfPoint> each = contours.iterator();
            markers = 0;
            //Loop through all contours and find max contour
            while (each.hasNext()) {
                MatOfPoint currentContour = each.next();

                double myArea = Imgproc.contourArea(currentContour);
                if (myArea > maxArea) {
                    maxArea = myArea;
                    maxContour = currentContour;
                }
                //Log.i(TAG, "Area=" + myArea);
            }

            if (contours.size() > 0) {
                double maxContourArea = Imgproc.contourArea(maxContour);
                Rect myRect = Imgproc.boundingRect(maxContour);
                // Log.i(TAG, "Marker Area =" + maxContourArea);
                if (maxContourArea > 500) {
                    float ratio = (float) myRect.height / (float) myRect.width;
                    Log.i(TAG, "Marker Width=" + myRect.width + " " + "Height=" + myRect.height);
                    Log.i(TAG, "Marker ratio=" + ratio);
                    if (ratio > 0.60)
                        markers = 4;
                    else markers = 1;
                    // Core.multiply(maxContour, new Scalar(4, 4), maxContour);
                    // mContours.add(maxContour);
                }
            }
            Log.i(TAG, "Markers=" + markers);

            if (markers != 0) {
                Rect rect = Imgproc.boundingRect(maxContour);
                Imgproc.rectangle(source0, rect, new Scalar(0.0, 0.0, 255.0), 2);
            }

            //saveMatToDisk(source0, "FinalFrame");

            bmp = Bitmap.createBitmap(source0.cols(), source0.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(source0, bmp);

            // just process one image;
            countMarkers = false;
            countMarkersReady = true;
        } // end of process image

        return source0;
    }

    public void countMarkers() {
        countMarkers = true;
    }

    public boolean countMarkersReady() {
        if (countMarkersReady) {
            countMarkersReady = false;
            return true;
        } else {
            return false;
        }
    }

    public int getNumMarkers(){
        return markers;
    }

    public Bitmap getBitmap() {
        return bmp;
    }
}
