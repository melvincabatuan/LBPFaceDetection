#include "io_github_melvincabatuan_lbpfacedetection_MainActivity.h"
#include <android/log.h>
#include <android/bitmap.h>
#include <stdlib.h>
#include <stdio.h>

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

#define  LOG_TAG    "LBPFaceDetection"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  DEBUG 1


/** Global variables */
char face_cascade_path[100];
char eyes_cascade_path[100]; 
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

double t; // measuring performance


/*
 * Class:     io_github_melvincabatuan_lbpfacedetection_MainActivity
 * Method:    predict
 * Signature: (Landroid/graphics/Bitmap;[B)V
 */
JNIEXPORT void JNICALL Java_io_github_melvincabatuan_lbpfacedetection_MainActivity_predict
  (JNIEnv * pEnv, jobject clazz, jobject pTarget, jbyteArray pSource){

   AndroidBitmapInfo bitmapInfo;
   uint32_t* bitmapContent; // Links to Bitmap content

   if(AndroidBitmap_getInfo(pEnv, pTarget, &bitmapInfo) < 0) abort();
   if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) abort();
   if(AndroidBitmap_lockPixels(pEnv, pTarget, (void**)&bitmapContent) < 0) abort();

   /// Access source array data... OK
   jbyte* source = (jbyte*)pEnv->GetPrimitiveArrayCritical(pSource, 0);
   if (source == NULL) abort();

   /// cv::Mat for YUV420sp source and output BGRA 
    Mat srcGray(bitmapInfo.height, bitmapInfo.width, CV_8UC1, (unsigned char *)source);
    Mat mbgra(bitmapInfo.height, bitmapInfo.width, CV_8UC4, (unsigned char *)bitmapContent);

/***********************************************************************************************/
    /// Native Image Processing HERE... 
    if(DEBUG){
      LOGI("Starting native image processing...");
    }


    if (face_cascade.empty()){
       t = (double)getTickCount();
       sprintf( face_cascade_path, "%s/%s", getenv("ASSETDIR"), "lbpcascade_frontalface.xml");       
    
      /* Load the face cascades */
       if( !face_cascade.load(face_cascade_path) ){ 
           LOGE("Error loading face cascade"); 
           abort(); 
       };

       t = 1000*((double)getTickCount() - t)/getTickFrequency();
       if(DEBUG){
       LOGI("Loading face cascade took %lf milliseconds.", t);
     }
    }
 


   
    if (eyes_cascade.empty()){
       t = (double)getTickCount();
       sprintf( eyes_cascade_path, "%s/%s", getenv("ASSETDIR"), "haarcascade_eye_tree_eyeglasses.xml");      
    
      /* Load the face cascades */
       if( !eyes_cascade.load(eyes_cascade_path) ){ 
           LOGE("Error loading eyes cascade"); 
           abort(); 
       };

       t = 1000*((double)getTickCount() - t)/getTickFrequency();
       if(DEBUG){
       LOGI("Loading eyes cascade took %lf milliseconds.", t);
       }
    }
    
           
 
     std::vector<Rect> faces;

/*
     //-- Equalize faces
     t = (double)getTickCount();
     equalizeHist( srcGray, srcGray);
     t = 1000*((double)getTickCount() - t)/getTickFrequency();
     if(DEBUG){
       LOGI("equalizeHist() time = %lf milliseconds.", t);
     }
*/

       //-- Detect faces
       t = (double)getTickCount();
       face_cascade.detectMultiScale( srcGray, faces, 1.1, 2, 0 , Size(80, 80) );
       t = 1000*((double)getTickCount() - t)/getTickFrequency();
       if(DEBUG){
       LOGI("face_cascade.detectMultiScale() time = %lf milliseconds.", t);
      }


       // Iterate through all faces and detect eyes
       t = (double)getTickCount();

       for( size_t i = 0; i < faces.size(); i++ )
       {
        Mat faceROI = srcGray( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        if( eyes.size() == 2)
        {
            //-- Draw the face
            Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
            ellipse( srcGray, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 255, 255 ), 2, 8, 0 );

            for( size_t j = 0; j < eyes.size(); j++ )
            { //-- Draw the eyes
                Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                circle( srcGray, eye_center, radius, Scalar( 255, 255, 255 ), 3, 8, 0 );
            }//endfor
        }//endif
      }//endfor
  
     t = 1000*((double)getTickCount() - t)/getTickFrequency();
     if(DEBUG){
         LOGI("Iterate through all faces and detecting eyes took %lf milliseconds.", t);
     }

   /// Display to Android
     cvtColor(srcGray, mbgra, CV_GRAY2BGRA);


    if(DEBUG){
      LOGI("Successfully finished native image processing...");
    }
   
/************************************************************************************************/ 
   
   /// Release Java byte buffer and unlock backing bitmap
   pEnv-> ReleasePrimitiveArrayCritical(pSource,source,0);
   if (AndroidBitmap_unlockPixels(pEnv, pTarget) < 0) abort();

}
