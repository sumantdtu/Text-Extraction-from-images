#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include<opencv/cv.h>
#include<fstream>

using namespace cv;
using namespace std;

#define INPUT_FILE              "D:/pic/test.jpg"
#define OUTPUT_FOLDER_PATH      string("D:/pic/min/")

int main(int argc, char* argv[])
{
    Mat im = imread(INPUT_FILE,1);
  // Mat rgb;
    // transpose(im,im);
       //   flip(im,im,1);

         // transpose(im,im);
           //    flip(im,im,1);

 //    transpose(im,im);

   //  flip(im,im,1);
    // downsample and use it for processing
    //pyrUp(im, rgb);
    Mat small,hsv;
   // cvtColor(im, hsv, COLOR_BGR2HSV);
    cvtColor(im, small, CV_BGR2GRAY);
    // morphological gradient
    Mat grad;
    Mat morphKernel=  getStructuringElement(MORPH_ELLIPSE, Size(15,20));

    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel); //important
          //  cv::namedWindow("mor", cv::WINDOW_AUTOSIZE);
        //cv::imshow("mor", grad);


    // binarize
    Mat bw;
  threshold(grad, bw, 0.0, 100.0, THRESH_BINARY | THRESH_OTSU);
    // connect horizontally oriented regions
    //cv::namedWindow("binarize", cv::WINDOW_AUTOSIZE);
    cv::imshow("binarize", bw);
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(2,2));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);

    cv::namedWindow("connt", cv::WINDOW_AUTOSIZE);
    cv::imshow("connt", connected);
    // find contours
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0,0));


   // cv::namedWindow("contour", cv::WINDOW_AUTOSIZE);
    //cv::imshow("contour", hierarchy);
    // filter contours
 int i=1; int count = 0;
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
    {
        Rect rect = boundingRect(contours[idx]);
        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
                  Mat roi = im(rect);

        // fill the contour
        drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
        // ratio of non-zero pixels in the filled region
        double r = (double)countNonZero(maskROI)/(rect.width*rect.height);


        if (r > .45 /* assume at least 45% of the area is filled if it contains text */
            &&
            (rect.height > 8 && rect.width > 8) /* constraints on region size */
            /* these two conditions alone are not very robust. better to use something
            like the number of significant peaks in a horizontal projection as a third condition */
            )
        {
            std::string savingName = OUTPUT_FOLDER_PATH + format("%d",i) + ".jpg";
cv::imwrite(savingName, roi);

      rectangle(im,rect,Scalar(255,0,255),1);

     //        namedWindow("rect", cv::WINDOW_AUTOSIZE);
            imshow("rect",im);
            waitKey(1000);
 //std::vector<size> rect;
//           imwrite(OUTPUT_FOLDER_PATH + format('%d'."jpg",i),roi);
      // imwrite(OUTPUT_FOLDER_PATH + string(i+"k.jpg"),roi);
     // i++;
     // imwrite((OUTPUT_FOLDER_PATH, "image_%04i.jpg" %i), roi)
    //i += 1;

//count++;
//i++;



   // for(int j=1;j<=count;)
    //{
       // cv::Mat ldim = cv::imread(OUTPUT_FOLDER_PATH + format("%d",j) + ".jpg", 0);

       // cv::Mat img = cv::imread(OUTPUT_FOLDER_PATH + format("%d",j) + ".jpg", 0);
        Mat img;
    pyrUp(roi, img);


  threshold(img, img, 225, 255, THRESH_BINARY);

  cv::bitwise_not(img, img);

  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3));
  cv::erode(img, img, element);

  std::vector<cv::Point> points;
  cv::Mat_<uchar>::iterator it = img.begin<uchar>();
  cv::Mat_<uchar>::iterator end = img.end<uchar>();
  for (; it != end; ++it)
    if (*it)
      points.push_back(it.pos());
  cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));

        double angle = box.angle;
  if (angle < -45.)
    angle += 90.;

      cv::Point2f vertices[4];
  box.points(vertices);
 // for(int k = 0; k < 4; ++k)
   // cv::line(img, vertices[k], vertices[(k + 1) % 4], cv::Scalar(255, 0, 0), 1, CV_AA);
  std::cout << "File " << OUTPUT_FOLDER_PATH + format("%d",i) + ".jpg" << ": " << angle << std::endl;
  cv::imshow("Result", img);
  cv::waitKey(2000);
  cv::Mat rot_mat = cv::getRotationMatrix2D(box.center, angle, 1);

  cv::Mat rotated;
  cv::warpAffine(img, rotated, rot_mat, img.size(), cv::INTER_CUBIC);

   // cvtColor(rotated, im, CV_GRAY2BGR);
     Mat grad;
    Mat morphKernel=  getStructuringElement(MORPH_ELLIPSE, Size(5, 3));

    morphologyEx(rotated, grad, MORPH_DILATE, morphKernel);
  std::string savingNam = OUTPUT_FOLDER_PATH + string("new") + format("%d",i) + ".jpg";
cv::imwrite(savingNam, rotated);
  cv::imshow("Rotated", grad);
  cv::waitKey(1000);
i++;

    }
        }
   /* String folderpath = "D:/pic/min/*.jpg";
vector<String> filenames;
cv::glob(folderpath, filenames);

for (size_t i=0; i<filenames.size(); i++)
{
    Mat ldim = imread(filenames[i]);
        imshow("new",ldim);

    waitKey(1000);
}
    */     //namedWindow("new", cv::WINDOW_AUTOSIZE);


   //imwrite(OUTPUT_FOLDER_PATH + string("8rgb.jpg"), im);
   // namedWindow("Output", cv::WINDOW_AUTOSIZE);
    //imshow("Output",im);
    //waitKey(0);

    return 0;
}
